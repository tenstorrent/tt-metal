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
    "pad": "K not a multiple of 32: the in0 reader zero-fills each row tile of the last K block (barrier + RISC-V loop)",
    "msync": "mcast receivers ack only after computing on a block: the sender's ack collection adds to every compute step",
    "mcrate": "multicast data moves at its own rate, falling with the receiver count (measured: one-to-all microbenchmark)",
    "corecc": "a core's DRAM read rate when other cores read DRAM concurrently (measured: 4-core interleaved reads)",
    "reusesync": "Reuse cores never synchronise, so only part of their per-step link load coincides (fitted fraction: 0.72-0.75, stable across folds and chips)",
    "dramw": "DRAM writes get their own (fitted) chip efficiency: bursty block-end writes sharing DRAM with other cores' reads (isolated 64-core writes reach 0.47)",
    "shardhop": "sharded in0: each K block's mcast sender is the core holding that slice, so every step pays a sender handoff",
}
EXPERIMENTAL = {
    "linkmc": "multicast traffic has its own link efficiency (link_eff then describes read traffic)",
    "linkbank": "link loads from only the DRAM banks each K step touches (bank camping concentrates link traffic)",
    "msync0": "in0 mcast only: receivers ack after computing on the block, so the ack collection adds to each compute step",
    "wlink": "output writes from every writer core to the DRAM banks load the NoC links; the most-loaded link bounds the write",
    "wov2d": "2D: most cores only receive in1, so their writer overlaps the next block's pipeline (only the excess is exposed)",
    "wburst": "DRAM writes congest with the bytes each writer has in flight per barrier (one subblock), like L1 read bursts",
    "mcovl": "the async multicast write overlaps the sender's next fetch; only the handshake is serial with it",
    "mcout": "MultiCore: a fixed cost per output tile (dest acquire, pack, one-tile write and its barrier)",
    "blkfix": "a fixed cost per output block: buffer handshakes, compute reconfiguration and output setup each block pays",
    "burstdram": "DRAM reads also congest with the bytes each reader has in flight per barrier (super-linear per-step cost)",
}
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
    "mc_eff0": (
        0.9,
        "multicast rate as a fraction of the link rate, extrapolated to zero receivers (fitted; the one-to-all microbenchmark's 63-receiver point, 0.43, matches the fit)",
        "mcrate",
    ),
    "mc_eff_rx": (0.0026, "drop in that fraction per receiver", "mcrate"),
    "noc_eff_cc": (
        0.48,
        "one core's DRAM read rate with other cores reading concurrently, fraction of 32 B/cycle",
        "corecc",
    ),
    "reuse_link": (
        0.5,
        "fraction of the lockstep link load that Reuse's unsynchronised cores actually put on a link at once",
        "reusesync",
    ),
    "wburst_KB": (20.0, "KB per writer per barrier at which its share of DRAM write bandwidth halves", "wburst"),
    "mc_out": (500.0, "cycles per MultiCore output tile: acquire, pack, write + barrier", "mcout"),
    "blk_fixed": (1000.0, "cycles per output block: CB handshakes, compute reconfig, output block setup", "blkfix"),
    "dram_w_eff": (0.6, "achievable fraction of spec DRAM bandwidth for writes", "dramw"),
    "burst_dram_KB": (500.0, "KB per reader per barrier at which its DRAM read rate halves", "burstdram"),
    "lat_shard": (
        500.0,
        "cycles per K step: handing the in0 mcast to the core that holds the next K slice",
        "shardhop",
    ),
    "pad_elem": (5.0, "cycles per element the in0 reader zero-fills in a partial K tile", "pad"),
    "link_eff_mc": (0.6, "achievable fraction of a NoC link's rate for multicast traffic", "linkmc"),
}
# measured on the device, held fixed in the fit: name -> ({arch: value}, source). Measured on WH only so far; BH
# values are fitted until measured there (a WH measurement is not a BH constant).
PINNED = {
    "dram_eff": ({"wh": 0.77}, "DRAM read bandwidth microbenchmark, 222 of 288 GB/s"),
    "launch_us": ({"wh": 0.5}, "profiler: kernel start to end minus the K loops, instrumented matmul kernels"),
    "rl_init": ({"wh": 340.0}, "profiler: spill path per subblock-step, instrumented compute kernel"),
    "u2d": ({"wh": 100.0}, "profiler: fp32 unpack-to-dest handshake per tile"),
    "lat_dram": (
        {"wh": 352.0},
        "data-movement microbenchmark: one core, N interleaved DRAM pages, one barrier (intercept)",
    ),
    "lat_l1": (
        {"wh": 282.0},
        "data-movement microbenchmark: one core, N interleaved L1 pages, one barrier (intercept)",
    ),
    "issue": ({"wh": 38.0}, "data-movement microbenchmark: per-page cost of small interleaved DRAM reads"),
    "noc_eff": ({"wh": 0.96}, "data-movement microbenchmark: one core's interleaved read rate, 30.7 of 32 B/cycle"),
    "noc_eff_cc": ({"wh": 0.48}, "multi-core interleaved DRAM reads: 15.4 B/cycle per core with 4 cores reading"),
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
LO = {
    "mc_eff0": 0.05,
    "mc_eff_rx": 1e-5,
    "dram_eff": 0.2,
    "noc_eff": 0.1,
    "l1_frac": 0.05,
    "link_eff": 0.05,
    "bank_frac": 0.02,
    "link_eff_mc": 0.05,
}
HI = {
    "mc_eff0": 1.0,
    "mc_eff_rx": 0.01,
    "dram_eff": 1.0,
    "noc_eff": 1.0,
    "l1_frac": 3.0,
    "link_eff": 1.0,
    "bank_frac": 1.0,
    "link_eff_mc": 1.0,
}


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
    # ttnn's output dtype defaults to in0's when the op doesn't name one
    g["tb_o"] = d.out_dtype.fillna(d.a_dtype).map(TILE_BYTES).fillna(2048).to_numpy(float)
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
    kr = d.K.to_numpy(float) % 32
    g["kpad"] = np.where(kr > 0, 32 * (32 - kr), 0.0)  # elements zero-filled per partial last-K tile of in0
    if "link_bytes" in d:
        g["link"] = d.link_bytes.to_numpy(float)
    if "bank_a" in d:
        g["bk_a"], g["bk_b"] = d.bank_a.to_numpy(float), d.bank_b.to_numpy(float)
    if "wlink" in d:
        g["wlink"] = d.wlink.to_numpy(float)
    if "link_mcast" in d:
        g["link_mc"] = d.link_mcast.to_numpy(float)
    return g


def annotate(d):
    """Add the per-row NoC link load and DRAM banks touched that geometry() reads (cached per layout)."""
    if on("link"):
        import nocload

        if on("linkbank"):
            import bankload

            d["link_bytes"] = nocload.link_bytes_banked(geometry(d), d, bankload.patterns(geometry(d), d))
        else:
            d["link_bytes"] = nocload.link_bytes(geometry(d), d)
        if on("wlink"):
            d["wlink"] = nocload.write_link(geometry(d), d)
        if on("linkmc"):
            d["link_mcast"] = nocload.link_bytes(geometry(d), d, part="mcast")
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
        noc_rd = noc
        if on("corecc"):  # concurrent DRAM readers each get the loaded per-core rate
            noc_rd = np.where(r > 1, s["noc_Bpc"] * p["noc_eff_cc"], noc)
        if on("sat"):
            cgd = 1 + nbytes / (p["burst_dram_KB"] * 1e3) if on("burstdram") else 1.0
            dr = np.maximum(p["lat_dram"] + nbytes * cgd / noc_rd, nbytes * r * cgd / (dram * eff))
            l1 = np.maximum(p["lat_l1"] + nbytes / noc, nbytes * r * cg / l1bw)
        else:
            dr = p["lat_dram"] + nbytes / np.minimum(noc, dram * eff / r)
            l1 = p["lat_l1"] + nbytes / np.minimum(noc, l1bw / (r * cg))
        return np.select([src == 0, src == 1], [np.maximum(dr, floor), np.maximum(l1, floor)], 0.0)

    def mcast(nbytes, rx):
        rate = noc
        if on("mcrate"):  # measured: 17.1 B/cycle to 24 receivers, 13.9 to 63 (WH)
            rate = s["noc_Bpc"] * np.maximum(p["mc_eff0"] - p["mc_eff_rx"] * rx, 0.05)
        if not on("mcast"):
            return np.where(rx > 0, nbytes / rate, 0.0)
        return np.where(rx > 0, p["lat_mcast"] + rx * p["ack_rx"] + nbytes / rate, 0.0)

    if on("mcovl"):  # sender: fetch block k+1 while block k's multicast data is still in flight

        def path(nbytes, src, rd, eff, cg, tile, rx):
            f = fetch(nbytes, src, rd, eff, cg, tile)
            m = mcast(nbytes, rx)
            hs = np.where(rx > 0, p["lat_mcast"] + rx * p["ack_rx"], 0.0) if on("mcast") else 0.0
            return np.maximum(f, m - hs) + hs

        step0 = path(b0, g["src_a"], g["rd0"], ea, cg0, g["tb_a"], g["rx0"])
        step1 = path(b1, g["src_b"], g["rd1"], eb, cg1, g["tb_b"], g["rx1"])
    else:
        step0 = fetch(b0, g["src_a"], g["rd0"], ea, cg0, g["tb_a"]) + mcast(b0, g["rx0"])  # in0 path (sender)
        step1 = fetch(b1, g["src_b"], g["rd1"], eb, cg1, g["tb_b"]) + mcast(b1, g["rx1"])  # in1 path (sender)
    da, db = g["rd0"] * b0 * (g["src_a"] == 0), g["rd1"] * b1 * (g["src_b"] == 0)  # chip DRAM bytes per K step
    la, lb = g["rd0"] * b0 * (g["src_a"] == 1), g["rd1"] * b1 * (g["src_b"] == 1)  # chip interleaved-L1 bytes
    chip = np.maximum((da + db) / dram, (la * cg0 + lb * cg1) / l1bw)
    if on("bank"):
        chip = np.maximum(chip, np.maximum(da / (dram * ea), db / (dram * eb)))
    read = np.maximum(np.maximum(step0, step1), chip)
    if on("link"):
        lk = g["link"] * (np.where(g["fam"] == "reuse", p["reuse_link"], 1.0) if on("reusesync") else 1.0)
        read = np.maximum(read, lk / (s["noc_Bpc"] * p["link_eff"]))
        if on("linkmc"):  # the most multicast-loaded link at the multicast efficiency
            read = np.maximum(read, g["link_mc"] / (s["noc_Bpc"] * p["link_eff_mc"]))

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
        cw = 1 + sbh * sbw * g["tb_o"] / (p["wburst_KB"] * 1e3) if on("wburst") else 1.0
        dram_w = s["dram_GBs"] * 1e9 * p["dram_w_eff"] / clk if on("dramw") else dram
        rate_w = np.minimum(noc, dram_w / (np.maximum(g["cores"], 1) * cw))
        write = np.select(
            [g["dst_o"] == 0, g["dst_o"] == 1],
            [nsb * p["lat_write"] + wbytes / rate_w, nsb * p["lat_l1"] + wbytes / noc],
            0.0,
        )
        if on("wlink"):  # every writer core writes its block at once: the busiest link carries wlink x one core's bytes
            write = np.maximum(
                write, np.where(g["dst_o"] == 0, wbytes * g["wlink"] / (s["noc_Bpc"] * p["link_eff"]), 0.0)
            )

    if on("mcast") and (on("msync0") or on("msync")):  # the sender gathers each receiver's ack after it computed
        sync = g["rx0"] * p["ack_rx"] + (g["rx1"] * p["ack_rx"] if on("msync") else 0.0)
        comp = comp + sync
    if on("shardhop"):  # rotating in0 sender: a cross-core handoff on the critical path of every K step
        comp = comp + np.where((g["src_a"] == 2) & (g["rx0"] > 0), p["lat_shard"], 0.0)
    piped = read + (nK - 1) * np.maximum(read, comp) + comp  # double-buffered K loop
    serial = nK * (read + comp)
    if on("wov2d"):  # 2D receivers: writing block b overlaps block b+1's reads and compute
        loop = np.where(g["dbuf"], piped, serial)
        block = loop + epi + np.where(g["fam"] == "2d", np.maximum(write - loop, write / np.maximum(nsb, 1)), write)
    else:
        block = np.where(g["dbuf"], piped, serial) + epi + write
    if on("blkfix"):
        block = block + p["blk_fixed"]
    pad = 0.0
    if on("pad"):  # last K block: per in0 row tile, a read barrier then the zero fill, serial on the in0 reader
        lat0 = np.select([g["src_a"] == 0, g["src_a"] == 1], [p["lat_dram"], p["lat_l1"]], 0.0)
        pad = np.where(g["kpad"] > 0, obh * (lat0 + g["kpad"] * p["pad_elem"]), 0.0)
        block = block + pad
    core = g["nob"] * block

    # MultiCore: one output tile at a time, every input tile read with its own barrier
    mc = g["fam"] == "multicore"
    tiles_pc = np.ceil(g["B"] * g["Mt"] * g["Nt"] / np.maximum(g["cores"], 1))
    tb2 = g["tb_a"] + g["tb_b"]  # one in0 tile and one in1 tile per K step, each read with its own barrier
    mc_read = np.maximum(g["Kt"] * (2 * p["tile_rt"] + tb2 / noc), g["Kt"] * tb2 * g["cores"] / dram)
    mc_tile = np.maximum(mc_read, g["Kt"] * 16.0 * g["ph"])
    mc_pad = np.where(g["kpad"] > 0, g["kpad"] * p["pad_elem"], 0.0) if on("pad") else 0.0
    mc_fixed = p["mc_out"] if on("mcout") else p.get("lat_write", 0.0)
    core = np.where(mc, tiles_pc * (mc_tile + mc_pad + mc_fixed), core)

    t = p["launch_us"] * 1e3 + core / clk * 1e9
    if parts:
        return t, dict(read=read, comp=comp, reload=reload, epi=epi, write=write, nob=g["nob"], nK=nK)
    return t
