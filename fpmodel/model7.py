"""First-principles device-time model for matmul program configs (v7).

Structure follows the kernels on main (metal2 forks; see the factory/kernel files under
ttnn/cpp/ttnn/operations/matmul/device/). Per core, for each output block (obh x obw tiles) the kernels step through
nK = Kt/kb K blocks:
  readers   NCRISC (in0) and BRISC (in1 + output writer) fetch one K block: DRAM / interleaved-L1 page reads, all in
            flight, one barrier; mcast senders then wait for receiver acks, multicast the block and set a flag.
  compute   per subblock (sbh x sbw): kb matmul_block calls; each does sbh*sbw tile products (16 cycles per fidelity
            phase); pack sbh*sbw tiles (partials or output). Partials are reloaded (spill/reload) when there are
            several K blocks and no packer L1 accumulation.
  overlap   input buffers hold two K blocks when B*nK > 1, so reads of block k+1 overlap compute of block k;
            otherwise read and compute serialise. The output writer shares BRISC with the in1 reader.
Time = launch + per-core blocks x block time, the critical core being an mcast sender (it reads and multicasts).

Each mechanism below is a TERM; ABLATE=term,... switches terms off (and drops their constants from the fit). PINNED
constants are measured on the device and held fixed; the rest are fitted (fit.py) by within-problem ranking error.
"""
import os, numpy as np

TERMS = {
    "link": "most-loaded NoC link per K step (nocload.py) caps the read rate",
    "bank": "a K step whose DRAM pages sit in a few interleaved banks gets only those banks' bandwidth (bankload.py)",
    "sat": "read latency is hidden once the shared DRAM / L1 bandwidth saturates (else latency + transfer)",
    "burst": "interleaved-L1 reads congest with the bytes each reader has in flight per barrier",
    "spill": "partials spill/reload: re-init per reloaded subblock, fp32 unpack-to-dest per tile",
    "mcast": "mcast sender handshake per K block plus one ack per receiver",
    "pack": "packer throughput per subblock",
    "call": "fixed cost per matmul_block call",
    "write": "output block write (BRISC, serialised with the in1 reader)",
    "epilogue": "bias add and SFPU activation per output tile",
    "issue": "each page read costs the reader core a fixed issue time (per-core floor on a K block's read)",
}
EXPERIMENTAL = {}
OFF = set(filter(None, os.environ.get("ABLATE", "").split(",")))
EXTRA = set(filter(None, os.environ.get("EXTRA", "").split(",")))  # experimental terms switched on
on = lambda t: (t in EXTRA) if t in EXPERIMENTAL else (t not in OFF)

TILE_BYTES = {"bf16": 2048, "bfp8": 1088, "bfp4": 576, "fp32": 4096}
PHASES = {"LoFi": 1, "HiFi2": 2, "HiFi3": 3, "HiFi4": 4}
SPEC = {
    "wh": dict(clk=1.0e9, noc_Bpc=32.0, dram_GBs=288.0),
    "bh": dict(clk=1.35e9, noc_Bpc=64.0, dram_GBs=512.0),
}

# constant: (initial value, unit/meaning, term or None for the core model)
CONSTANTS = {
    "dram_eff": (0.77, "achievable fraction of spec DRAM bandwidth", None),
    "noc_eff": (0.7, "achievable fraction of the per-core NoC read/write rate", None),
    "l1_frac": (0.8, "chip interleaved-L1 read bandwidth as a fraction of the DRAM spec", None),
    "lat_dram": (500.0, "cycles: one batch of DRAM page reads, issue to barrier", None),
    "lat_l1": (250.0, "cycles: one batch of interleaved-L1 page reads", None),
    "launch_us": (0.5, "us: program launch + kernel prologue/epilogue", None),
    "tile_rt": (500.0, "cycles: MultiCore factory, one tile read with its own barrier", None),
    "link_eff": (0.7, "achievable fraction of a NoC link's rate", "link"),
    "bank_frac": (0.2, "DRAM bandwidth one bank serves, as a fraction of the chip's", "bank"),
    "burst_KB": (600.0, "KB per reader per barrier at which interleaved-L1 bandwidth halves", "burst"),
    "rl_init": (340.0, "cycles per reloaded subblock: MOP reprogram + drains", "spill"),
    "u2d": (100.0, "cycles per reloaded fp32 tile: unpack-to-dest handshake", "spill"),
    "lat_mcast": (30.0, "cycles: mcast sender handshake per K block", "mcast"),
    "ack_rx": (15.0, "cycles per mcast receiver ack", "mcast"),
    "pack_Bpc": (25.0, "packer bytes per cycle", "pack"),
    "call": (30.0, "cycles per matmul_block call", "call"),
    "lat_write": (65.0, "cycles: one subblock of output page writes + barrier", "write"),
    "sfpu_tile": (2000.0, "cycles per output tile of SFPU activation", "epilogue"),
    "issue": (100.0, "cycles per page read issued by one reader core", "issue"),
}
# measured on the device, held fixed in the fit: name -> ({arch: value}, source). Measured on WH only so far; BH
# values are fitted until measured there (a WH measurement is not a BH constant).
PINNED = {
    "dram_eff": ({"wh": 0.77}, "DRAM read bandwidth microbenchmark, 222 of 288 GB/s"),
    "launch_us": ({"wh": 0.5}, "profiler: kernel start to end minus the K loops, instrumented matmul kernels"),
    "rl_init": ({"wh": 340.0}, "profiler: spill path per subblock-step, instrumented compute kernel"),
    "u2d": ({"wh": 100.0}, "profiler: fp32 unpack-to-dest handshake per tile"),
}


def pin(fit_module, arch):
    """hold this arch's measured constants fixed in fit_module's bounds (restoring the defaults for the others)"""
    for k, (vals, _) in PINNED.items():
        if k not in PARAMS:
            continue
        if arch in vals:
            v = vals[arch]
            fit_module.LO[k], fit_module.HI[k] = v * 0.999, v * 1.001
        else:
            fit_module.LO[k], fit_module.HI[k] = LO.get(k, 1e-3), HI.get(k, 1e7)


PARAMS = {k: v[0] for k, v in CONSTANTS.items() if v[2] is None or on(v[2])}
LO = {"dram_eff": 0.2, "noc_eff": 0.1, "l1_frac": 0.05, "link_eff": 0.05, "bank_frac": 0.02}
HI = {"dram_eff": 1.0, "noc_eff": 1.0, "l1_frac": 3.0, "link_eff": 1.0, "bank_frac": 1.0}


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
    is2d, in0, in1, reuse = (fam == f for f in ("2d", "1d_in0", "1d_in1", "reuse"))
    R2, C2 = np.ceil(Mrows / pcM), np.ceil(Nt / pcN)  # 2D: a fused batch stacks its rows
    P0, P1 = np.ceil(Nt / pcN), np.ceil(Mrows / pcM)
    grid = (d.grid_x * d.grid_y).fillna(d.cores).to_numpy(float)
    # Reuse: output blocks of per_core_M x per_core_N over the batch-stacked rows, spread over cores; a block taller
    # than one batch (per_core_M > Mt) is walked as per_core_M / Mt blocks of Mt rows (the factory's batch_scale_factor)
    bsf = np.where(pcM > Mt, np.floor(pcM / Mt), 1.0)
    rblocks = np.ceil(B * Mt / pcM) * np.ceil(Nt / pcN)
    rcores = np.minimum(rblocks, grid)
    g["cores"] = np.select([is2d, in0, in1, reuse], [R2 * C2, P0, P1, rcores], d.cores.to_numpy(float))
    g["rx0"] = np.select([is2d, in0], [C2 - 1, P0 - 1], 0.0)  # in0 multicast receivers
    g["rx1"] = np.select([is2d, in1], [R2 - 1, P1 - 1], 0.0)  # in1 multicast receivers
    g["rd0"] = np.select([is2d, in0, in1, reuse], [R2, 1, P1, rcores], 0.0)  # cores reading in0 from memory
    g["rd1"] = np.select([is2d, in0, in1, reuse], [C2, P0, 1, rcores], 0.0)  # cores reading in1 from memory
    obh = np.where(reuse, np.minimum(pcM, Mt), obh)
    obw = np.where(reuse, pcN, obw)
    g["nob"] = np.where(reuse, np.ceil(rblocks / rcores) * bsf, bloop * np.ceil(pcM / obh) * np.ceil(pcN / obw))
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
    g["dst_o"] = src(d.out_mem.to_numpy())
    g["arch"] = d.arch_.to_numpy()
    if "link_bytes" in d:
        g["link"] = d.link_bytes.to_numpy(float)
    if "bank_a" in d:
        g["bk_a"], g["bk_b"] = d.bank_a.to_numpy(float), d.bank_b.to_numpy(float)
    return g


def annotate(d):
    """Add the per-row NoC link load and DRAM banks touched that geometry() reads (cached per layout)."""
    if on("link"):
        import nocload

        d["link_bytes"] = nocload.link_bytes(geometry(d), d)
    if on("bank"):
        import bankload

        d["bank_a"], d["bank_b"], _ = bankload.banks_touched(geometry(d), d)
    return d


def predict(g, p, parts=False):
    """Device time in ns. p: constants for one architecture; g from geometry() for that arch."""
    s = SPEC[g["arch"][0]]
    clk = s["clk"]
    noc = s["noc_Bpc"] * p["noc_eff"]  # bytes per cycle per core
    dram = s["dram_GBs"] * 1e9 * p["dram_eff"] / clk  # chip DRAM bytes per cycle
    l1bw = s["dram_GBs"] * 1e9 * p["l1_frac"] / clk  # chip interleaved-L1 bytes per cycle
    kb, obh, obw, sbh, sbw, nsb, nK = g["kb"], g["obh"], g["obw"], g["sbh"], g["sbw"], g["nsb"], g["nK"]
    b0, b1 = obh * kb * g["tb_a"], kb * obw * g["tb_b"]  # bytes of one K block of in0, in1

    # ---- reads per K block ----
    ea = np.minimum(1.0, g["bk_a"] * p["bank_frac"]) if on("bank") else 1.0  # share of DRAM bandwidth reachable
    eb = np.minimum(1.0, g["bk_b"] * p["bank_frac"]) if on("bank") else 1.0
    cg0 = 1 + b0 / (p["burst_KB"] * 1e3) if on("burst") else 1.0  # interleaved-L1 congestion
    cg1 = 1 + b1 / (p["burst_KB"] * 1e3) if on("burst") else 1.0

    def fetch(nbytes, src, readers, eff, cg, tile):
        """one K block read by one core: all pages in flight, then a barrier; bandwidth shared by `readers` cores"""
        r = np.maximum(readers, 1)
        floor = nbytes / tile * p["issue"] if on("issue") else 0.0
        if on("sat"):
            dr = np.maximum(p["lat_dram"] + nbytes / noc, nbytes * r / (dram * eff))
            l1 = np.maximum(p["lat_l1"] + nbytes / noc, nbytes * r * cg / l1bw)
        else:
            dr = p["lat_dram"] + nbytes / np.minimum(noc, dram * eff / r)
            l1 = p["lat_l1"] + nbytes / np.minimum(noc, l1bw / (r * cg))
        return np.select([src == 0, src == 1], [np.maximum(dr, floor), np.maximum(l1, floor)], 0.0)

    def mcast(nbytes, rx):
        if not on("mcast"):
            return np.where(rx > 0, nbytes / noc, 0.0)
        return np.where(rx > 0, p["lat_mcast"] + rx * p["ack_rx"] + nbytes / noc, 0.0)

    step0 = fetch(b0, g["src_a"], g["rd0"], ea, cg0, g["tb_a"]) + mcast(b0, g["rx0"])  # in0 path (sender)
    step1 = fetch(b1, g["src_b"], g["rd1"], eb, cg1, g["tb_b"]) + mcast(b1, g["rx1"])  # in1 path (sender)
    da, db = g["rd0"] * b0 * (g["src_a"] == 0), g["rd1"] * b1 * (g["src_b"] == 0)  # chip DRAM bytes per K step
    la, lb = g["rd0"] * b0 * (g["src_a"] == 1), g["rd1"] * b1 * (g["src_b"] == 1)  # chip interleaved-L1 bytes
    chip = np.maximum((da + db) / dram, (la * cg0 + lb * cg1) / l1bw)
    if on("bank"):
        chip = np.maximum(chip, np.maximum(da / (dram * ea), db / (dram * eb)))
    read = np.maximum(np.maximum(step0, step1), chip)
    if on("link"):
        read = np.maximum(read, g["link"] / (s["noc_Bpc"] * p["link_eff"]))

    # ---- compute per K block: per subblock max(math, pack) across the TRISCs, plus fixed costs ----
    math = kb * sbh * sbw * 16.0 * g["ph"]
    pack = sbh * sbw * g["tb_p"] / p["pack_Bpc"] if on("pack") else 0.0
    fixed = kb * p["call"] if on("call") else 0.0
    tiles = obh * obw
    reload = 0.0
    if on("spill"):  # compute-side work inside the K loop, so it pipelines against the reads
        reload = g["reloads"] * (nsb * p["rl_init"] + np.where(g["tb_p"] >= 4096, tiles * p["u2d"], 0.0))
    comp = nsb * (np.maximum(math, pack) + fixed) + reload / np.maximum(nK, 1)
    epi = 0.0
    if on("epilogue"):
        epi = np.where(g["bias"], tiles * g["tb_o"] / p.get("pack_Bpc", 1e9), 0.0)
        epi = epi + np.where(g["sfpu"], tiles * p["sfpu_tile"], 0.0)

    # ---- output write ----
    write = 0.0
    if on("write"):
        wbytes = tiles * g["tb_o"]
        rate_w = np.minimum(noc, dram / np.maximum(g["cores"], 1))
        write = np.select(
            [g["dst_o"] == 0, g["dst_o"] == 1],
            [nsb * p["lat_write"] + wbytes / rate_w, nsb * p["lat_l1"] + wbytes / noc],
            0.0,
        )

    piped = read + (nK - 1) * np.maximum(read, comp) + comp  # double-buffered K loop
    serial = nK * (read + comp)
    block = np.where(g["dbuf"], piped, serial) + epi + write
    core = g["nob"] * block

    # MultiCore: one output tile at a time, every input tile read with its own barrier
    mc = g["fam"] == "multicore"
    tiles_pc = np.ceil(g["B"] * g["Mt"] * g["Nt"] / np.maximum(g["cores"], 1))
    tb2 = g["tb_a"] + g["tb_b"]  # one in0 tile and one in1 tile per K step, each read with its own barrier
    mc_read = np.maximum(g["Kt"] * (2 * p["tile_rt"] + tb2 / noc), g["Kt"] * tb2 * g["cores"] / dram)
    mc_tile = np.maximum(mc_read, g["Kt"] * 16.0 * g["ph"])
    core = np.where(mc, tiles_pc * (mc_tile + p.get("lat_write", 0.0)), core)

    t = p["launch_us"] * 1e3 + core / clk * 1e9
    if parts:
        return t, dict(read=read, comp=comp, reload=reload, epi=epi, write=write, nob=g["nob"], nK=nK)
    return t
