"""First-principles device-time model for matmul program configs.

Structure follows the kernels on main (metal2 forks; see the factory/kernel files under
ttnn/cpp/ttnn/operations/matmul/device/). Per core, for each output block (obh x obw tiles) the kernels step through
nK = Kt/kb K blocks:
  readers   NCRISC (in0) and BRISC (in1 + output writer) fetch one K block: DRAM / interleaved-L1 page reads, all in
            flight, one barrier; mcast senders then wait for receiver acks, multicast the block and set a flag.
  compute   per subblock (sbh x sbw): kb matmul_block calls; each unpacks sbh in0 + sbw in1 tiles and does sbh*sbw tile
            products (16 cycles per fidelity phase); pack sbh*sbw tiles (partials or output). Partials are reloaded
            (spill/reload) when there are several K blocks and no packer L1 accumulation.
  overlap   input buffers hold two K blocks when B*nK > 1, so reads of block k+1 overlap compute of block k;
            otherwise read and compute serialise. The output writer shares BRISC with the in1 reader, so writing an
            output block delays the next block's first read.
Time = launch + per-core blocks x block time, the critical core being an mcast sender (it reads and multicasts).
Constants are hardware rates (spec) and per-event latencies; see PARAMS.
"""
import os, numpy as np, pandas as pd

MX = set(filter(None, os.environ.get("MX", "").split(",")))  # experimental mechanisms switched on

TILE_BYTES = {"bf16": 2048, "bfp8": 1088, "bfp4": 576, "fp32": 4096}
PHASES = {"LoFi": 1, "HiFi2": 2, "HiFi3": 3, "HiFi4": 4}

# Spec values (fixed) and initial guesses for the event costs (fitted or not, see fit.py).
SPEC = {
    "wh": dict(clk=1.0e9, noc_Bpc=32.0, dram_GBs=288.0),
    "bh": dict(clk=1.35e9, noc_Bpc=64.0, dram_GBs=512.0),
}
PARAMS = {  # name: initial value; cycles unless noted
    "dram_eff": 0.75,  # achievable fraction of spec DRAM bandwidth
    "noc_eff": 0.7,  # achievable fraction of the per-core NoC read/write rate
    "l1_frac": 0.8,  # chip-wide interleaved-L1 read bandwidth as a fraction of the DRAM spec (both are NoC-limited)
    "burst_KB": 1000.0,  # congestion: shared read bandwidth / (1 + per-core bytes per barrier / burst_KB)
    "lat_dram": 1000.0,  # one batch of DRAM page reads: issue + round trip to the barrier
    "lat_l1": 400.0,  # one batch of interleaved-L1 page reads
    "lat_mcast0": 600.0,  # in0 mcast handshake (NCRISC, NOC1): wait for acks, multicast, flag
    "lat_mcast1": 600.0,  # in1 mcast handshake (BRISC, NOC0)
    "ack_rx": 5.0,  # per receiver: the sender's semaphore collects one remote increment per receiver
    "lat_write": 600.0,  # one subblock of output page writes + write barrier
    "unpack_Bpc": 40.0,  # unpacker bytes per cycle (srcA/srcB fill)
    "pack_Bpc": 40.0,  # packer bytes per cycle
    "call": 30.0,  # per matmul_block call (inner k) fixed cost
    "subblock": 150.0,  # per subblock: dest acquire/commit/wait/release + pack setup
    "reload": 100.0,  # per reloaded subblock: reconfig + copy init
    "sfpu_tile": 400.0,  # SFPU activation per output tile (gelu/silu/...)
    "launch_us": 4.0,  # program launch + kernel prologue/epilogue (microseconds)
    "tile_rt": 1500.0,  # MultiCore: one tile read with a barrier after it (latency-bound round trip)
}
if (
    "rl" in MX
):  # reload re-init (MOP reprogram + drains) per reloaded subblock, and fp32 unpack-to-dest handshake per tile
    PARAMS.update(rl_init=200.0, u2d=100.0)
if "dram" in MX:  # DRAM efficiency limited by bytes in flight across readers (one K block per reader between barriers)
    PARAMS.update(eta_max=0.85, inflight_KB=500.0)
if "noc" in MX:  # most-loaded NoC link per K step (nocload.py), at link_eff of the link rate
    PARAMS.update(link_eff=0.7)
if "bank" in MX:  # DRAM bandwidth one interleaved bank can serve, as a fraction of the chip's (bankload.py)
    PARAMS.update(bank_frac=0.12)
if "launch" in MX:  # per-core dispatch cost
    PARAMS.update(launch_core_ns=20.0)


def cdiv(a, b):
    return -(-a // b)


def geometry(d):
    """Per-row config and problem quantities, as float arrays."""
    g = {}
    t = lambda x: np.ceil(x.to_numpy(float) / 32)
    Mt, Kt, Nt, B = t(d.M), t(d.K), t(d.N), d.batch.to_numpy(float)
    fam = d.family.to_numpy()
    fuse = d.fuse_batch.fillna(0).to_numpy() == 1
    pcM, pcN = d.per_core_M.to_numpy(float), d.per_core_N.to_numpy(float)
    obh = np.where(np.isnan(d.out_block_h), pcM, d.out_block_h).astype(float)
    obw = np.where(np.isnan(d.out_block_w), pcN, d.out_block_w).astype(float)
    kb = d.in0_block_w.fillna(1).to_numpy(float)
    sbh, sbw = d.out_subblock_h.fillna(1).to_numpy(float), d.out_subblock_w.fillna(1).to_numpy(float)
    Mrows = np.where(fuse, B * Mt, Mt)
    bloop = np.where(fuse, 1.0, B)  # outer batch loop on each core
    is2d, in0, in1, reuse, mc = (fam == f for f in ("2d", "1d_in0", "1d_in1", "reuse", "multicore"))
    # in0 / in1 mcast receivers per sender, cores working, output blocks per core
    R2, C2 = np.ceil(Mt / pcM), np.ceil(Nt / pcN)
    P0, P1 = np.ceil(Nt / pcN), np.ceil(Mrows / pcM)
    grid = (d.grid_x * d.grid_y).fillna(d.cores).to_numpy(float)
    rblocks = B * np.ceil(Mt / pcM) * np.ceil(Nt / pcN)  # Reuse: whole (pcM x pcN) blocks spread over cores
    rcores = np.minimum(rblocks, grid)
    g["cores"] = np.select([is2d, in0, in1, reuse], [R2 * C2, P0, P1, rcores], d.cores.to_numpy(float))
    g["rx0"] = np.select([is2d, in0], [C2 - 1, P0 - 1], 0.0)  # in0 multicast receivers
    g["rx1"] = np.select([is2d, in1], [R2 - 1, P1 - 1], 0.0)  # in1 multicast receivers
    g["rd0"] = np.select([is2d, in0, in1, reuse], [R2, 1, P1, rcores], 0.0)  # cores reading in0 from memory
    g["rd1"] = np.select([is2d, in0, in1, reuse], [C2, P0, 1, rcores], 0.0)  # cores reading in1 from memory
    obh = np.where(reuse, pcM, obh)
    obw = np.where(reuse, pcN, obw)
    g["nob"] = np.where(reuse, np.ceil(rblocks / rcores), bloop * np.ceil(pcM / obh) * np.ceil(pcN / obw))
    g["nK"] = np.ceil(Kt / kb)
    g["dbuf"] = np.where(reuse, True, bloop * g["nK"] > 1)
    g.update(Mt=Mt, Kt=Kt, Nt=Nt, B=B, kb=kb, obh=obh, obw=obw, sbh=sbh, sbw=sbw, fam=fam)
    g["nsb"] = np.ceil(obh / sbh) * np.ceil(obw / sbw)
    g["tb_a"] = d.a_dtype.map(TILE_BYTES).to_numpy(float)
    g["tb_b"] = d.b_dtype.map(TILE_BYTES).to_numpy(float)
    g["tb_o"] = d.out_dtype.map(TILE_BYTES).fillna(2048).to_numpy(float)
    acc32 = d.fp32_acc.fillna(0).to_numpy() == 1
    l1acc = d.packer_l1_acc.fillna(0).to_numpy() == 1
    bias = d.bias.fillna(0).to_numpy() == 1
    g["tb_p"] = np.where(acc32, 4096.0, np.where(l1acc, 2048.0, g["tb_o"]))  # partials format
    nK = g["nK"]
    l1on = l1acc & np.where(bias, nK > 1, nK > 2)
    g["reloads"] = np.where(l1on, np.where(bias, 0.0, 1.0), np.maximum(nK - 1, 0))  # reload passes per output block
    g["bias"] = bias
    act = d.activation.fillna("").astype(str).to_numpy()
    g["sfpu"] = (act != "") & (act != "relu") & (act != "nan")
    g["ph"] = d.fidelity.map(PHASES).fillna(4).to_numpy(float)
    src = lambda m: np.select([m == "dram", m == "l1"], [0, 1], 2)  # 0 dram, 1 interleaved L1, 2 sharded
    g["src_a"], g["src_b"] = src(d.a_mem.to_numpy()), src(d.b_mem.to_numpy())
    om = d.out_mem.to_numpy()
    g["dst_o"] = src(om)
    g["arch"] = d.arch_.to_numpy()
    if "link_bytes" in d:
        g["link"] = d.link_bytes.to_numpy(float)
    if "bank_a" in d:
        g["bk_a"], g["bk_b"] = d.bank_a.to_numpy(float), d.bank_b.to_numpy(float)
    return g


def predict(g, p, parts=False):
    """Device time in ns. p: dict of PARAMS values (scalars) for one architecture; g from geometry() for that arch."""
    arch = g["arch"][0]
    s = SPEC[arch]
    clk = s["clk"]
    noc = s["noc_Bpc"] * p["noc_eff"]  # bytes per cycle per core
    dram = s["dram_GBs"] * 1e9 * p["dram_eff"] / clk  # chip DRAM bytes per cycle
    if "dram" in MX:
        b0_, b1_ = g["obh"] * g["kb"] * g["tb_a"], g["kb"] * g["obw"] * g["tb_b"]
        infl = g["rd0"] * b0_ * (g["src_a"] == 0) + g["rd1"] * b1_ * (g["src_b"] == 0)
        dram = s["dram_GBs"] * 1e9 * p["eta_max"] * np.minimum(1.0, infl / (p["inflight_KB"] * 1e3)) / clk
        dram = np.maximum(dram, 1e-3)
    kb, obh, obw, sbh, sbw, nsb = g["kb"], g["obh"], g["obw"], g["sbh"], g["sbw"], g["nsb"]
    b0, b1 = obh * kb * g["tb_a"], kb * obw * g["tb_b"]  # bytes of one K block of in0, in1

    l1bw = s["dram_GBs"] * 1e9 * p["l1_frac"] / clk  # chip interleaved-L1 bytes per cycle

    tot = g["rd0"] * b0 * (g["src_a"] < 2) + g["rd1"] * b1 * (g["src_b"] < 2)  # chip bytes in flight per K step

    ea = eb = 1.0
    if "bank" in MX:  # a K step whose pages sit in a few banks gets only those banks' share of the DRAM bandwidth
        ea = np.minimum(1.0, g["bk_a"] * p["bank_frac"])
        eb = np.minimum(1.0, g["bk_b"] * p["bank_frac"])

    def fetch(nbytes, src, readers, eff=1.0):
        """one K block read by one core, all pages in flight then a barrier; DRAM / interleaved L1 shared by `readers` cores"""
        cb = tot if "congtot" in MX else nbytes
        cg = 1 + cb / (p["burst_KB"] * 1e3)
        if "burstl1" in MX:
            cg = np.where(src == 1, cg, 1.0)  # congestion only on interleaved-L1 reads
        r = np.maximum(readers, 1) * (1 if "congchip" in MX else cg)
        if "sat" in MX:  # latency hides under the other readers' transfers once the shared bandwidth saturates
            return np.select(
                [src == 0, src == 1],
                [
                    np.maximum(p["lat_dram"] + nbytes / noc, nbytes * r / (dram * eff)),
                    np.maximum(p["lat_l1"] + nbytes / noc, nbytes * r / l1bw),
                ],
                0.0,
            )
        return np.select(
            [src == 0, src == 1],
            [
                p["lat_dram"] + nbytes / np.minimum(noc, dram * eff / r),
                p["lat_l1"] + nbytes / np.minimum(noc, l1bw / r),
            ],
            0.0,
        )

    mcast = lambda nbytes, rx, lat: np.where(rx > 0, lat + rx * p["ack_rx"] + nbytes / noc, 0.0)
    step0 = fetch(b0, g["src_a"], g["rd0"], ea) + mcast(b0, g["rx0"], p["lat_mcast0"])  # in0 path per K block (sender)
    step1 = fetch(b1, g["src_b"], g["rd1"], eb) + mcast(b1, g["rx1"], p["lat_mcast1"])  # in1 path per K block (sender)
    cong0, cong1 = (
        (1 + tot / (p["burst_KB"] * 1e3),) * 2
        if "congtot" in MX
        else (1 + b0 / (p["burst_KB"] * 1e3), 1 + b1 / (p["burst_KB"] * 1e3))
    )
    dc0, dc1 = (1.0, 1.0) if "burstl1" in MX else (cong0, cong1)
    chip = np.maximum(
        (g["rd0"] * b0 * dc0 * (g["src_a"] == 0) + g["rd1"] * b1 * dc1 * (g["src_b"] == 0))
        / dram,  # all readers at once
        (g["rd0"] * b0 * cong0 * (g["src_a"] == 1) + g["rd1"] * b1 * cong1 * (g["src_b"] == 1)) / l1bw,
    )
    if "bank" in MX:
        chip = np.maximum(
            chip,
            np.maximum(
                g["rd0"] * b0 * (g["src_a"] == 0) / (dram * ea), g["rd1"] * b1 * (g["src_b"] == 0) / (dram * eb)
            ),
        )
    read = np.maximum(np.maximum(step0, step1), chip)
    if "noc" in MX:
        read = np.maximum(read, g["link"] / (s["noc_Bpc"] * p["link_eff"]))

    # compute per K block: per subblock max(unpack, math, pack) pipelined across TRISCs, plus fixed costs
    math = kb * sbh * sbw * 16.0 * g["ph"]
    unpack = kb * (sbh * g["tb_a"] + sbw * g["tb_b"]) / p["unpack_Bpc"]
    if "unp" in MX:  # the smaller subblock side is held in srcA/srcB; the other operand is re-unpacked once per pair
        held_a = sbw >= sbh
        unpack = (
            kb
            * np.where(
                held_a,
                sbh * g["tb_a"] + np.ceil(sbh / 2) * sbw * g["tb_b"],
                sbw * g["tb_b"] + np.ceil(sbw / 2) * sbh * g["tb_a"],
            )
            / p["unpack_Bpc"]
        )
    pack = sbh * sbw * g["tb_p"] / p["pack_Bpc"]
    tiles = obh * obw
    # spill/reload of partials is compute-side (unpacker) work inside the K loop, so it pipelines against the reads
    reload = g["reloads"] * (tiles * g["tb_p"] / p["unpack_Bpc"] + nsb * p["reload"])
    if "rl" in MX:
        reload = reload + g["reloads"] * (nsb * p["rl_init"] + np.where(g["tb_p"] >= 4096, tiles * p["u2d"], 0.0))
    comp = nsb * (np.maximum(np.maximum(math, unpack), pack) + kb * p["call"] + p["subblock"]) + reload / np.maximum(
        g["nK"], 1
    )
    epi = np.where(
        g["bias"], tiles * (g["tb_p"] / p["unpack_Bpc"] + g["tb_o"] / p["pack_Bpc"]) + nsb * p["subblock"], 0.0
    )
    epi = epi + np.where(g["sfpu"], tiles * p["sfpu_tile"], 0.0)

    # output block write (BRISC, serialised with the in1 reader): one barrier per subblock
    wbytes = tiles * g["tb_o"]
    rate_w = np.minimum(noc, dram / np.maximum(g["cores"], 1))
    write = np.select(
        [g["dst_o"] == 0, g["dst_o"] == 1],
        [nsb * p["lat_write"] + wbytes / rate_w, nsb * p["lat_l1"] + wbytes / noc],
        0.0,
    )

    nK = g["nK"]
    piped = read + (nK - 1) * np.maximum(read, comp) + comp  # double-buffered K loop
    serial = nK * (read + comp)
    if "wov" in MX:  # the writer drains each output subblock as compute packs it: only the part of the write that
        # outlasts the last K step's remaining subblocks is exposed, plus the last subblock's write
        last = comp * (nsb - 1) / np.maximum(nsb, 1)
        write = np.maximum(write - last, write / np.maximum(nsb, 1))
    block = np.where(g["dbuf"], piped, serial) + epi + write
    core = g["nob"] * block

    # MultiCore: one output tile at a time, every input tile read with its own barrier
    mc = g["fam"] == "multicore"
    tiles_pc = np.ceil(g["B"] * g["Mt"] * g["Nt"] / np.maximum(g["cores"], 1))
    mc_tile = np.maximum(
        g["Kt"] * 2 * (p["tile_rt"] + 2048 / noc), g["Kt"] * np.maximum(16.0 * g["ph"], 2 * 2048 / p["unpack_Bpc"])
    )
    core = np.where(mc, tiles_pc * (mc_tile + p["lat_write"]), core)

    t = p["launch_us"] * 1e3 + core / clk * 1e9
    if "launch" in MX:
        t = t + p["launch_core_ns"] * g["cores"]
    if parts:
        return t, dict(read=read, comp=comp, reload=reload, epi=epi, write=write, nob=g["nob"], nK=nK)
    return t
