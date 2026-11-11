import re
"""E1a: run gdn_spec_step's own Metal2 kernels (prefixed + lever-1 path: K4, SFPU gates, VB 4) as a CLASSIC generic_op,
so the GDN role binaries of the engine can host the unchanged step code (spec §17).

On BH a Metal2 DataflowBuffer is the classic CB interface, so the op's kernel sources compile as-is once they see:
  - namespace dfb: DFBBindingToken{cb id, LLKMetadata{format}} for every name they reference (unused names alias a
    configured CB of the same format, they are never touched in this variant);
  - namespace tensor: TensorBindingToken<CTA offset, CRTA byte offset> (accessor CTAs + addresses as common RTAs);
  - namespace sem: SemaphoreBindingToken{id, LOCAL_NONATOMIC} (program semaphores 0, 1);
  - a kernel_main() that calls the TT_KERNEL entry with the named compile-time values and per-core runtime args.
The host side mirrors GdnSpecStepProgramFactory for this variant (DFB sizes / formats / UnpackToDest, defines, args).
CB ids stay < 50 (the AR sender uses >= 50 on the combined cores)."""
import os, struct
import ttnn

F32, B16, U32 = 0, 5, 24   # tt::DataFormat codes in LLKMetadata
FMT = {F32: ttnn.float32, B16: ttnn.bfloat16, U32: ttnn.uint32}
# the step kernel sources: E1_KDIR = an absolute path into the run's PRIVATE tree (the shared in-pod tree stays stock r13;
# a relative path would resolve against the shared tree's include root)
KDIR = os.environ.get("E1_KDIR", "")
assert KDIR.startswith("/"), "set E1_KDIR to the private kernels dir (absolute)"
bits = lambda f: struct.unpack("<I", struct.pack("<f", f))[0]


# ctrl page bytes (B + B*Nv u32 = 64 words since v1.13): the ctrl_r / ctrl_w CBs must hold the WHOLE page. At 64 B the
# reader's 256 B ctrl read overran ctrl_r by 192 B into the next CB (tmp: rows 0..2 of its first tile, the compute's
# squares for the RMS norm) -- harmless while the next layer's rd_ctrl ran before the compute's c_D, an out-only
# mismatch (one core's block, rows 0..2) once SP_LATE let the compute reach c_D first
CTRL_B = 256
# engine GDN role (build_engine, lean=True): DFBs dropped from the table and bound to another id instead (spec v1.50:
# frees 3 CB ids on the GDN cores for gdn_out). ctrl_w / gnew -> gprev: writer-local, never pushed across a gprev use
# (ctrl_w's reserve/push/pop runs at the writer's layer start, before wr_rcoef stages into gprev; gnew is dead code
# under GDN_BATCH); eps_l2 -> eps_norm: never generated / waited under GDN_NO_EPS_L2 (PREFIXED: the prefix did the l2)
LEAN_ALIAS = {"ctrl_w": "gprev", "gnew": "gprev", "eps_l2": "eps_norm",
              "linv": "inv"}  # linv lives in WY (reserve..pop), inv in c_D: disjoint, compute-local, same fp32 tile


def dfb_table(Kt, Vt, T, VB, out_b16=True, batch=False, st_ring=0, lean=False, kmq_users=4):
    """(name, entries, bytes, fmt) of every DFB the PREFIXED + PREFIXED_GATES + LAZY + VSPLIT + SFPU_GATE program uses.
    lean (engine role): LEAN_ALIAS entries dropped; kmq_users = per-user K_u / Q_u slots in the kmq ring (lean: 2)."""
    tab = _dfb_table(Kt, Vt, T, VB, out_b16, batch, st_ring, kmq_users)
    return [e for e in tab if e[0] not in LEAN_ALIAS] if lean else tab


def _dfb_table(Kt, Vt, T, VB, out_b16, batch, st_ring, kmq_users):
    Vs, t32, t16 = Vt // VB, 4096, 2048
    KVs, Ch = Kt * Vs, 2 * Kt + Vt
    NBm = 8 if batch else 1  # GDN_BATCH: per-user slots for every user of the head (kBatchMax = 8)
    return [
        # reader-produced
        ("w_in", Vt, t16, B16), ("scaler", 1, t32, F32), ("eps_l2", 1, t16, B16), ("eps_norm", 1, t16, B16),
        # GDN_ST_RING (batch): the state / stash slots are a ring of st_ring users (streamed per user)
        ("state_in", (st_ring or NBm) * KVs, t32, F32), ("ktp_in", (st_ring or NBm) * Kt, t32, F32),
        ("up_in", (st_ring or NBm) * Vs, t32, F32),
        ("knq", 2 * Kt, t32, F32), ("vc", Vs, t32, F32), ("zs", Vt, t32, F32), ("ctrl_r", 1, CTRL_B, U32),
        # compute-local
        ("tmp", max(Kt, Vt), t32, F32), ("stats", 1, t32, F32), ("scratch", 1, t32, F32), ("inv", 1, t32, F32),
        ("kt", Kt, t32, F32), ("gq", Vt, t32, F32), ("on", Vt, t32, F32), ("sp", NBm * KVs, t32, F32),
        ("ksqs", 2 * Vs, t32, F32), ("kkqk", 2, t32, F32), ("npo", 4 if T > 4 else 3, t32, F32),
        ("aa", 3 if T > 4 else 2, t32, F32), ("linv", 1, t32, F32), ("rhs", Vs, t32, F32), ("u", Vs, t32, F32),
        ("wu", Vs, t32, F32),
        # compute-produced
        ("out", Vs, t16 if out_b16 else t32, B16 if out_b16 else F32), *([] if batch else [("dnew", KVs, t32, F32)]),
        ("kt_out", (NBm * ((Kt + VB - 1) // VB)) if batch else Kt, t32, F32), ("u_out", Vs, t32, F32), ("acc_mine", Vs, t32, F32),
        # writer
        ("ctrl_w", 1, CTRL_B, U32), ("bounce", Ch, t16, B16), ("gprev", 1, t32, F32), ("gnew", 1, t32, F32),
        ("acc_x", Vt, t32, F32), ("coef", 15 if batch else 8, t32, F32),
    ] + ([("l12", 1, t32, F32)] if T > 4 else []) + ([("kmq", kmq_users * Kt, t32, F32), ("kts", NBm * ((Kt + VB - 1) // VB), t32, F32),
                                                # stg: [U_u (8 * Vs) | the decay pages of users u = s mod VB]
                                                ("stg", NBm * Vs + (NBm + VB - 1) // VB, t32, F32), ("spd", 1, 64, U32)] if batch else [])


# names the three kernels reference but this variant never touches -> alias of a configured CB of the same format
ALIAS = {"src_in": "w_in", "taps": "w_in", "z_in": "w_in", "sel": "w_in", "mask_T": "w_in", "wc": "w_in",
         "shift": "w_in", "wout": "w_in", "ab_w": "w_in", "cv": "tmp", "qc": "tmp", "kc": "tmp", "gt": "tmp",
         "dtb_ws": "tmp", "nea_ws": "tmp", "gsc": "ctrl_w", "e_t": "w_in", "acc_a": "tmp"}
# the factory builds the DM kernels at Os (kernel config buffer); the classic descriptor has no opt-level field here,
# so the wrapper asks GCC for size optimisation on everything it compiles after this line
DM_OPT = '#pragma GCC optimize ("Os")\n' if os.environ.get("E1_DM_OS", "1") == "1" else ""
RT_BM = os.environ.get("E1_RT_BM", "1") == "1"
GPX = os.environ.get("E1_GPX", "")  # size diff: absolute dir of the prefix agent's gpx headers, stubbed in (never fires)   # B / M runtime (GDN_RT_BM): one binary for B = 1 and B = 8
UNPACK_TO_DEST = ("state_in", "up_in", "coef", "ksqs", "kkqk", "vc")

READER_TENSORS = ["qkv", "win_a", "win_b", "ring", "ctrl", "taps", "dtb", "nea", "weight", "pre"]
WRITER_TENSORS = ["ring_out", "out", "win_a_w", "win_b_w", "ctrl_w", "pre_w", "qkv_w", "dtb_w", "nea_w"]


def _bind_header(ids, tensors, cta_offs, crta0=0):
    L = ['#pragma once', '#include "internal/template_string.h"', '#include "api/dataflow/dataflow_buffer.h"',
         '#include "api/dataflow/semaphore_binding_token.h"', '#include "api/tensor/tensor_binding_token.h"',
         'struct ScratchpadBindingToken;', 'namespace dfb {']
    for name, (cb, fmt) in ids.items():
        L.append(f'constexpr DFBBindingToken<dfb::AccessPattern::STRIDED, dfb::AccessPattern::STRIDED> {name}{{{cb}, '
                 f'::binding_details::LLKMetadata{{.format = {fmt}u, .face_r_dim = 16u, .face_c_dim = 16u, '
                 f'.num_faces_r_dim = 2u, .num_faces_c_dim = 2u}}}};')
    L += ['}  // namespace dfb', 'namespace sem {',
          'constexpr ::SemaphoreBindingToken ready{0u, ::SemScope::LOCAL_NONATOMIC};',
          'constexpr ::SemaphoreBindingToken valid{1u, ::SemScope::LOCAL_NONATOMIC};', '}  // namespace sem',
          'namespace tensor {']
    for i, t in enumerate(tensors):
        L.append(f'using {t}_t = ::tensor_accessor::TensorBindingToken<{cta_offs[i]}u, {4 * (crta0 + i)}u>;')
        L.append(f'constexpr {t}_t {t}{{::binding_details::LLKMetadata{{.format = 0u, .face_r_dim = 16u, '
                 f'.face_c_dim = 16u, .num_faces_r_dim = 2u, .num_faces_c_dim = 2u}}}};')
    L += ['}  // namespace tensor']
    return "\n".join(L) + "\n"


def build(dev, cores, g, B, T, Kt, Vt, VB, tens, kopts, scale, l2_eps, norm_eps, gen_dir, pre_s, out_b16=True,
          batch=False, layers=None):
    """batch: GDN_BATCH (type 4 batched): core (h, s) runs all B users of head h; len(cores) == Nv * VB.
    layers: GDN_LOOP -- a list of per-layer tensor dicts (same keys / shapes as tens); the step entry runs once per layer
    inside one kernel, the tensor addresses (common runtime args) rewritten before each call."""
    NL = len(layers) if layers else 1
    """tens: dict qkv win_a win_b ring ctrl taps dtb nea weight pre out (device tensors). Returns a ProgramDescriptor."""
    Nv, Nk, K = g.Nv, g.Nk, 4
    Lw, C = K - 1 + T, g.C
    Ct, M, Vs = C // 32, B, Vt // VB
    assert len(cores) == (Nv * VB if batch else B * Nv * VB)
    os.makedirs(gen_dir, exist_ok=True)
    tab = dfb_table(Kt, Vt, T, VB, out_b16, batch)
    ids = {n: (i, f) for i, (n, _, _, f) in enumerate(tab)}
    assert len(tab) < 50, len(tab)
    for a, b in ALIAS.items():
        if a not in ids:
            ids[a] = ids[b]
    if "dnew" not in ids:
        ids["dnew"] = ids["tmp"]  # GDN_BATCH: S' is written from sp (the name is still referenced, never used)
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    cbs = [ttnn.CBDescriptor(total_size=e * nb, core_ranges=crs, format_descriptors=[ttnn.CBFormatDescriptor(
        buffer_index=ids[n][0], data_format=FMT[f], page_size=nb)]) for n, e, nb, f in tab]

    def ctas(names):
        out, offs = [], []
        for t in names:
            offs.append(len(out))
            out += ttnn.TensorAccessorArgs(tens[t]).get_compile_time_args()
        return out, offs
    rmap = {"qkv": "qkv", "win_a": "win_a", "win_b": "win_b", "ring": "ring", "ctrl": "ctrl", "taps": "taps",
            "dtb": "dtb", "nea": "nea", "weight": "weight", "pre": "pre"}
    wmap = {"ring_out": "ring", "out": "out", "win_a_w": "win_a", "win_b_w": "win_b", "ctrl_w": "ctrl",
            "pre_w": "pre", "qkv_w": "qkv", "dtb_w": "dtb", "nea_w": "nea"}
    rct, roff = ctas([rmap[t] for t in READER_TENSORS])
    wct, woff = ctas([wmap[t] for t in WRITER_TENSORS])
    hr, hw, hc = (os.path.join(gen_dir, f"bind_{k}.h") for k in ("r", "w", "c"))
    open(hr, "w").write(_bind_header(ids, READER_TENSORS, roff))
    open(hw, "w").write(_bind_header(ids, WRITER_TENSORS, woff))
    open(hc, "w").write(_bind_header(ids, [], []))
    W = g.W // 32
    z_tile0, ab_page = (2 * Nk * Kt * 32 + Nv * Vt * 32) // 32, g.QKVZ // 32
    # one row-major uint32 page [1, N]: N*4 bytes (64 at B = 1, 256 at B = 8: 1 + B + B*Nv words padded to 16)
    ctrl_bytes = int(tens["ctrl"].shape[-1]) * 4
    assert ctrl_bytes <= CTRL_B, (ctrl_bytes, CTRL_B)  # the ctrl_r / ctrl_w CBs hold the whole ctrl page
    hs = 4294967295
    TB, TM = (1, 1) if RT_BM else (B, M)   # template B / M: fixed under GDN_RT_BM (the runtime values are args 4/5, 11/12)
    tr = os.path.join(gen_dir, "e1_reader.cpp")
    open(tr, "w").write(
        f'{DM_OPT}#include "{hr}"\nvoid kernel_main();\n'
        f'inline uint32_t gdn_rt_B() {{ return get_arg_val<uint32_t>(4); }}\n'
        f'inline uint32_t gdn_rt_M() {{ return get_arg_val<uint32_t>(5); }}\n'
        f'#include "{KDIR}/dataflow/reader_gdn_spec_step.cpp"\n'
        + (f'#include "{GPX}/gpx_dm.hpp"\n' if GPX else '') +
        f'void kernel_main() {{ volatile tt_l1_ptr uint32_t* crta = (volatile tt_l1_ptr uint32_t*)get_common_arg_addr(0);\n'
        f'  for (uint32_t L = 0; L < {NL}; ++L) {{\n'
        + (f'    for (uint32_t t = 0; t < {len(READER_TENSORS)}; ++t) crta[t] = get_arg_val<uint32_t>(8 + L * {len(READER_TENSORS)} + t);\n' if NL > 1 else '') +
        f'    reader<{Kt}, {Vt}, {Nk}, {Nv}, {T}, {TB}, {TM}, {K}, {Lw}, {Ct}, {z_tile0}, {ab_page}, {W}, '
        f'1, 1, {bits(l2_eps)}u, {bits(norm_eps)}u, {ctrl_bytes}, {hs}u, 0, {VB}>(get_arg_val<uint32_t>(0), '
        f'get_arg_val<uint32_t>(1), get_arg_val<uint32_t>(2), get_arg_val<uint32_t>(3));\n  }}\n'
        + (f'  if (get_arg_val<uint32_t>(6) == 0xC0FFEEu) {{ const uint32_t a = get_arg_val<uint32_t>(7); '
           f'gpx::Ctrl<{B}, {T}, {M}, {Nv}> c; c.parse(a); const auto wa = TensorAccessor(tensor::win_a); '
           f'const auto wb = TensorAccessor(tensor::win_b); gpx::read_window<{B}, {Ct}>(wa, wa, c, a, a); '
           f'gpx::stage_window_out<{B}, {T}, {K}>(c, a, a, a); noc_async_read_barrier(); }}\n' if GPX else '')
        + '}\n')
    tw = os.path.join(gen_dir, "e1_writer.cpp")
    open(tw, "w").write(
        f'{DM_OPT}#include "{hw}"\nvoid kernel_main();\n'
        f'inline uint32_t gdn_rt_B() {{ return get_arg_val<uint32_t>(11); }}\n'
        f'inline uint32_t gdn_rt_M() {{ return get_arg_val<uint32_t>(12); }}\n'
        f'#include "{KDIR}/dataflow/writer_gdn_spec_step.cpp"\n'
        + (f'#include "{GPX}/gpx_dm.hpp"\n' if GPX else '')
        + ('static uint32_t g_gdn_l = 0;\nuint32_t gdn_loop_k() { return g_gdn_l + 1; }\n' if NL > 1 else '') +
        f'void kernel_main() {{ volatile tt_l1_ptr uint32_t* crta = (volatile tt_l1_ptr uint32_t*)get_common_arg_addr(0);\n'
        f'  for (uint32_t L = 0; L < {NL}; ++L) {{\n'
        + (f'    g_gdn_l = L;\n    for (uint32_t t = 0; t < {len(WRITER_TENSORS)}; ++t) crta[t] = get_arg_val<uint32_t>(15 + L * {len(WRITER_TENSORS)} + t);\n' if NL > 1 else '') +
        f'    writer<{Kt}, {Vt}, {Nk}, {Nv}, {T}, {TB}, {TM}, {Lw}, {Ct}, {Nv * Vt}, {ctrl_bytes}, {hs}u, {VB}>('
        + ", ".join(f"get_arg_val<uint32_t>({i})" for i in range(11)) + ");\n  }\n"
        + (f'  if (get_arg_val<uint32_t>(13) == 0xC0FFEEu) {{ const uint32_t a = get_arg_val<uint32_t>(14); '
           f'gpx::Ctrl<{B}, {T}, {M}, {Nv}> c; c.parse(a); const auto wa = TensorAccessor(tensor::win_a_w); '
           f'const auto wb = TensorAccessor(tensor::win_b_w); gpx::stage_shift<{B}, {T}, {K}>(a, a); '
           f'gpx::write_window<{B}, {T}, {K}, {Ct}>(wa, wa, c, a, a, a); (void)wb; noc_async_write_barrier(); }}\n' if GPX else '')
        + "}\n")
    tc = os.path.join(gen_dir, "e1_compute.cpp")
    open(tc, "w").write(
        f'#include "{hc}"\nvoid kernel_main();\n'
        f'uint32_t gdn_rt_B();\n'
        f'#include "{KDIR}/compute/gdn_spec_step.cpp"\n'
        f'uint32_t gdn_rt_B() {{ return get_arg_val<uint32_t>(3); }}\n'
        + (f'#include "{GPX}/gpx_compute.hpp"\n' if GPX else '') +
        f'void kernel_main() {{ for (uint32_t L = 0; L < {NL}; ++L) {{ compute<{Kt}, {Vt}, {T}, {K}, 0, {bits(scale)}u, {bits(1.0 / (Vt * 32))}u, {VB}>('
        f'get_arg_val<uint32_t>(0), get_arg_val<uint32_t>(1), get_arg_val<uint32_t>(2)); }}\n'
        + ('  if (get_arg_val<uint32_t>(4) == 0xC0FFEEu) { const uint32_t c = get_arg_val<uint32_t>(5); '
           'gpx::conv<4, true>(c, c + 1, c + 2, c + 3); gpx::silu_z(c, c + 1); '
           'gpx::l2_root<4>(gpx::L2Cbs{c, c + 1, c + 2, c + 3, c + 4, c + 5, c + 6, c + 7, c + 8}, c); }\n' if GPX else '')
        + '}\n')
    pre_w = Nv * pre_s
    d_r = [("GDN_FAST", "1"), ("GDN_GATE_SCALARS", "1"), ("GDN_EARLY_SRC", "1"), ("GDN_LAZY", "1"),
           ("GDN_PREFIXED", "1"), ("GDN_PRE_W", str(pre_w)), ("GDN_PRE_S", str(pre_s))] + ([("GDN_RT_BM", "1")] if RT_BM else [])
    d_w = [("GDN_LAZY", "1"), ("GDN_SFPU_GATE", "1"), ("GDN_GATE_SCALARS", "1"), ("GDN_AB_PAGE", str(ab_page)),
           ("GDN_W_TILES", str(W)), ("GDN_DTB_FP32", "1"), ("GDN_NEA_FP32", "1"), ("GDN_NARROW_GR", "1"),
           ("GDN_VSPLIT", "1"), ("GDN_PREFIXED", "1"), ("GDN_PREFIXED_GATES", "1"), ("GDN_PRE_W", str(pre_w)),
           ("GDN_PRE_S", str(pre_s))] + ([("GDN_RT_BM", "1")] if RT_BM else [])
    if NL > 1:
        d_w.append(("GDN_LOOP", "1"))
    if batch:
        d_r.append(("GDN_BATCH", "1"))
        d_w.append(("GDN_BATCH", "1"))
    d_c = ([("GDN_BATCH", "1")] if batch else []) + [(k, "1") for k in ("GDN_FAST", "GDN_FP32_STATE", "GDN_GATE_SCALARS", "GDN_LAZY", "GDN_PREFIXED",
                              "GDN_PREFIXED_GATES", "GDN_SFPU_GATE", "GDN_SFPU_GATE_FACE", "GDN_VSPLIT", "GDN_VS_TAIL")]
    rrt, wrt, crt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for i, c in enumerate(cores):
        bh, s = i // VB, i % VB
        u, h = (0, bh) if batch else (bh // Nv, bh % Nv)
        peers = [dev.worker_core_from_logical_core(cores[bh * VB + p]) for p in range(VB)] + [None] * (4 - VB)
        pxy = []
        for p in peers:
            pxy += [p.x, p.y] if p is not None else [0, 0]
        rrt[c.x][c.y] = [u, h, 0, s, B, M, 0, 0] + (
            [lt[rmap[t]].buffer_address() for lt in layers for t in READER_TENSORS] if NL > 1 else [])
        wrt[c.x][c.y] = [u, h, s] + pxy + [B, M, 0, 0] + (
            [lt[wmap[t]].buffer_address() for lt in layers for t in WRITER_TENSORS] if NL > 1 else [])
        crt[c.x][c.y] = [u, 0, s, B, 0, 0]
    utd = [ttnn.UnpackToDestMode.Default] * 64
    for n in UNPACK_TO_DEST:
        utd[ids[n][0]] = ttnn.UnpackToDestMode.UnpackToDestFp32
    ccfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                        math_approx_mode=False, dst_full_sync_en=False)
    ccfg.unpack_to_dest_mode = utd
    ks = [ttnn.KernelDescriptor(kernel_source=tr, core_ranges=crs, compile_time_args=rct, defines=d_r,
                                common_runtime_args=[tens[rmap[t]].buffer_address() for t in READER_TENSORS],
                                runtime_args=rrt, config=ttnn.DataMovementConfigDescriptor(
                                    processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1)),
          ttnn.KernelDescriptor(kernel_source=tw, core_ranges=crs, compile_time_args=wct, defines=d_w,
                                common_runtime_args=[tens[wmap[t]].buffer_address() for t in WRITER_TENSORS],
                                runtime_args=wrt, config=ttnn.DataMovementConfigDescriptor(
                                    processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0)),
          ttnn.KernelDescriptor(kernel_source=tc, core_ranges=crs, compile_time_args=[], defines=d_c,
                                runtime_args=crt, config=ccfg)]
    sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=crs, initial_value=0) for i in range(2)]
    return ttnn.ProgramDescriptor(kernels=ks, semaphores=sems, cbs=cbs)


def gdn_role_cores(dev, n=24):
    """§17 placement: (0..9, 8..9) + (10..11, 8..9); falls back to the first n cores of a smaller grid."""
    gr = dev.compute_with_storage_grid_size()
    if gr.x >= 12 and gr.y >= 10:
        return [ttnn.CoreCoord(x, y) for y in (8, 9) for x in range(12)][:n]
    return [ttnn.CoreCoord(i % gr.x, i // gr.x) for i in range(n)]


# ---------------------------------------------------------------------------------------------------------------------
# G2: the engine GDN role (spec §17 / §20, v1.33). Same step kernels, looped over the pass's GDN layers, with:
#   * no per-core runtime args: everything per core (and the per-layer tensor addresses) in ONE arg block of kind gdn = 3
#     (engine/argblock.py layout: [magic, kind, n, version, payload..], 16 B padded; HEIGHT_SHARDED L1 over the GDN
#     cores, so its address is one common runtime arg, slot 3 of the engine's [phase, ar, v12, gdn, attn] common args);
#   * the head exchange on two device-local global semaphores vb0 / vb1 (GDN_EXT_SEM; no program semaphores);
#   * in_ready (raised +1 per GDN layer by the engine) gates the reader's `pre` reads (GDN_IN_WAIT) and the writer's
#     layer start; the state / stash reads go out first (GDN_STATE_FIRST);
#   * out_ready: +1 per core per layer on the engine's target core once the output rows landed (GDN_OUT_SIGNAL);
#   * at program end the writer resets in_ready / attn_kv / vb0 / vb1 on its core (it is their last waiter, v1.33).
AB_MAGIC, AB_KIND_GDN, AB_HDR, CRTA0 = 0x4D4B4142, 3, 4, 5
R_ARGS, W_ARGS, C_ARGS = 8, 15, 6


def argblock_tensor(dev, cores, payloads, kind=AB_KIND_GDN):
    """One core's block per row (engine/argblock.py layout), HEIGHT_SHARDED in L1 over `cores` (same address on every
    core / device). payloads: list of u32 lists, one per core in `cores` order."""
    import torch
    width = -(-(AB_HDR + max(len(p) for p in payloads)) // 4) * 4
    rows = [[AB_MAGIC, kind, len(p), 1] + [int(v) & 0xFFFFFFFF for v in p] for p in payloads]
    rows = [r + [0] * (width - len(r)) for r in rows]
    host = torch.tensor(rows, dtype=torch.int64).to(torch.int32).reshape(len(cores), width)
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                           ttnn.ShardSpec(crs, [1, width], ttnn.ShardOrientation.ROW_MAJOR))
    kw = {}
    if hasattr(dev, "get_num_devices") and dev.get_num_devices() > 1:
        kw["mesh_mapper"] = ttnn.ReplicateTensorToMesh(dev)
    return ttnn.from_torch(host, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=mc, **kw)


def _audit_cb_transfers(tab, layers, Kt, Vt, T, VB):
    """Static audit (v1.58 ask): every NoC transfer of the GDN role that lands in a CB fits the CB's page / slot.
    A 64 B ctrl_r CB under a 256 B ctrl read overran into tmp (the out-only race); this catches that class at build time."""
    import ttnn
    ent = {n: (e, nb) for n, e, nb, _ in tab}
    tile_b = lambda t: 32 * 32 * {ttnn.float32: 4, ttnn.bfloat16: 2, ttnn.uint32: 4}[t.dtype]
    Vs = Vt // VB
    for ly in layers:
        ctrl_b = int(ly["ctrl"].shape[-1]) * 4
        # rd_ctrl / wr ctrl: one whole ctrl page into ctrl_r / ctrl_w (lean: ctrl_w -> gprev, 4 KB)
        assert ctrl_b <= ent["ctrl_r"][1], ("ctrl_r", ctrl_b, ent["ctrl_r"])
        assert ctrl_b <= ent.get("ctrl_w", ent["gprev"])[1], ("ctrl_w", ctrl_b)
        # state / stash pages (ring, fp32 tiles) into state_in / ktp_in / up_in / gprev (decay page) / S' out of sp
        rb = tile_b(ly["ring"])
        for n in ("state_in", "ktp_in", "up_in", "gprev", "sp", "stg", "kt_out", "u_out"):
            assert rb == ent[n][1], (n, rb, ent[n])
        # pre tiles (knq | vc | zs | coefficient + gate pages) into knq / vc / zs / coef / gprev
        pb = tile_b(ly["pre"])
        for n in ("knq", "vc", "zs", "coef", "gprev"):
            assert pb == ent[n][1], (n, pb, ent[n])
        # the norm weight: row 0 (entry / 64 bytes per face row) of a bf16 tile into w_in
        assert tile_b(ly["weight"]) == ent["w_in"][1], ("w_in", tile_b(ly["weight"]), ent["w_in"])
        # out rows / whole out tiles from the out CB into the out tensor (same tile format)
        assert tile_b(ly["out"]) == ent["out"][1], ("out", tile_b(ly["out"]), ent["out"])
    # slot counts: the per-layer reservations fit the CBs
    assert ent["coef"][0] >= 15 and ent["kts"][0] >= 8 * ((Kt + VB - 1) // VB) and ent["kmq"][0] >= Kt, (ent["coef"], ent["kts"], ent["kmq"])
    assert ent["stg"][0] >= 8 * Vs + (8 + VB - 1) // VB and ent["acc_x"][0] >= Vt and ent["sp"][0] >= 8 * Kt * Vs


def build_engine(dev, cores, g, B, T, Kt, Vt, VB, layers, scale, l2_eps, norm_eps, gen_dir, pre_s, sems, out_target,
                 out_b16=True, kdir=None, st_ring=None, attn_n=0, skip_cb_ids=(34, 35, 37, 39), gdnout=None):
    """The GDN role's kernels / CBs / io for the engine program. layers: one tensor dict per GDN layer of the pass (keys
    as build's tens). sems: {in_ready, attn_kv, vb0, vb1, out_ready} L1 addresses; out_target: (noc x, noc y) of the
    core whose out_ready gets +1 per GDN core per layer. Core i of `cores` (row-major) = head i // VB, value block i % VB.
    Returns dict(kernels, cbs, io, info)."""
    kd = kdir or KDIR
    assert kd.startswith("/"), "the step kernels' dir must be absolute"
    NG, Nv, Nk, K = len(layers), g.Nv, g.Nk, 4
    assert NG >= 1 and len(cores) == Nv * VB, (NG, len(cores))
    Lw, C = K - 1 + T, g.C
    Ct, M = C // 32, B
    tens = layers[0]
    os.makedirs(gen_dir, exist_ok=True)
    # st_ring: users per state / stash ring (GDN_ST_RING; 0 = all 8 resident, the full early-state prefetch). Default
    # E1_ST_RING env, else 0. 2 = -216 KB per GDN core (needed only below an ~1.06 MB L1 floor on the GDN rows; QSK: 1.19 MB)
    st_ring = int(os.environ.get("E1_ST_RING", "0")) if st_ring is None else st_ring
    # GDN_EARLY_REPLAY (default on): the writer builds W / C_u before in_ready (its own in_ready wait, GDN_W_IN_WAIT),
    # the compute replays before the prefix inputs
    early = os.environ.get("E1_EARLY_REPLAY", "1") == "1"
    # GDN_SP_LATE (default on): S' leaves sp after the output rows (writer), the compute pops sp at the next layer's
    # replay / program end -- the S' DRAM write is off the step's critical path
    sp_late = os.environ.get("E1_SP_LATE", "1") == "1"
    # GDN_RCOEF_FAST (default on, needs early): wr_rcoef reads each user's decay rows (256 B) under one barrier and fills
    # C_u by NoC doubling instead of 1024 RISC stores per user (~36 us of writer time per layer at u8)
    rcoef_fast = os.environ.get("E1_RCOEF_FAST", "1") == "1"
    # GDN_SP_RD (needs sp_late): the reader (NOC1) writes S' after its kmq stream instead of the writer after the out rows
    # (default on with GDN_OUT_RD: the writer's 128 KB S' burst on NOC0 at +36 us otherwise jams the peers' acc_x exchange)
    sp_rd = sp_late and os.environ.get("E1_SP_RD", "1") == "1"
    # GDN_ZS_LATE (default on, needs early): the compute waits zs right before the gated output, not before the replay
    zs_late = early and os.environ.get("E1_ZS_LATE", "1") == "1"
    # GDN_CUT_FAST (default on): kts / kmq slots re-zero only the previous occupant's rows (one barrier per chunk)
    cut_fast = os.environ.get("E1_CUT_FAST", "1") == "1"
    # GDN_OUT_TILE (default on): at B * T == 32 the out tiles go out as whole tiles (1 write each, same bytes)
    out_tile = os.environ.get("E1_OUT_TILE", "1") == "1"
    # GDN_OUT_RD (E1_OUT_RD, B * T == 32 only): the reader (NOC1) writes the out tiles + raises the out signal instead of
    # the writer (NOC0 at physical x 1-4 is saturated by the weight stream when out leaves, v1.91). Same bytes / order.
    out_rd = os.environ.get("E1_OUT_RD", "1") == "1" and B * T == 32 and os.environ.get("E1_OUT_LOCAL", "0") in ("", "0") \
        and not gdnout   # MKA_GDNOUT: no out tensor write and its own out_ready (after gdn_out); GDN_OUT_RD does not apply
    rtens = READER_TENSORS + (["out_r"] if out_rd else [])
    # lean CB table (E1_LEAN, default on): 3 ids freed (LEAN_ALIAS) and kmq 4 -> 2 user slots (-32 KB)
    lean = os.environ.get("E1_LEAN", "1") == "1"
    tab = dfb_table(Kt, Vt, T, VB, out_b16, True, st_ring, lean=lean, kmq_users=2 if lean else 4)
    # CB ids 0..44 (spec v1.25 MAX_CB_ID), skipping the AR senders' ids on the corner cores (RS 34, RED 35, HDR 37,
    # AG 39: ArPhase.sender_parts) -- one table on all 24 GDN cores
    free_ids = [i for i in range(45) if i not in set(skip_cb_ids)]
    assert len(tab) <= len(free_ids), (len(tab), len(free_ids))
    ids = {n: (free_ids[i], f) for i, (n, _, _, f) in enumerate(tab)}
    for a, b in (list(LEAN_ALIAS.items()) if lean else []) + list(ALIAS.items()):  # LEAN first: ALIAS gsc -> ctrl_w
        if a not in ids:
            ids[a] = ids[b]
    ids.setdefault("dnew", ids["tmp"])
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    cbs = [ttnn.CBDescriptor(total_size=e * nb, core_ranges=crs, format_descriptors=[ttnn.CBFormatDescriptor(
        buffer_index=ids[n][0], data_format=FMT[f], page_size=nb)]) for n, e, nb, f in tab]
    rmap = {t: t for t in READER_TENSORS}
    rmap["out_r"] = "out"
    wmap = {"ring_out": "ring", "out": "out", "win_a_w": "win_a", "win_b_w": "win_b", "ctrl_w": "ctrl",
            "pre_w": "pre", "qkv_w": "qkv", "dtb_w": "dtb", "nea_w": "nea"}

    def ctas(names):
        out, offs = [], []
        for t in names:
            offs.append(len(out))
            out += ttnn.TensorAccessorArgs(tens[t]).get_compile_time_args()
        return out, offs
    # ---- MKA_GDNOUT (spec v1.8a): the GDN out_proj through the wo unit (attn/kernels/wo_unit_*.hpp), wrapper-only ----
    # per GDN layer on core i: ring CB 41 <- its gdn_out slice (BFP4, k-major 24 x nt pages, filled right after reader<>()
    # of the same layer: filling layer L + 1 there instead blocked the reader on the ring slot until gdn_out(L) and held
    # the next layer's state prefetch to the stall end, +50 us per GDN layer at u8); A CB 42 <- the layer's 24 `out`
    # tiles from DRAM once every GDN core raised GDNA; out = A @ slice -> CB 43 -> the owners' CB_XO chunks (rot order, wo
    # unit scatter, += count on the owners' word), then the engine's out_ready (the GDN-type layer stall ends after gdn_out)
    GO = gdnout
    go_r = go_w = go_c = go_sig = ""
    go_inc = ""
    if GO:
        WOU = GO["wou_dir"]
        RP, MAXT = GO["wk"] * GO["maxt"], GO["maxt"]
        CR_, CA_, CO_ = GO.get("cb", (42, 43, 44))   # ring, A, out slot (ids the role's table leaves free)
        _used = {ids[n][0] for n in ids}
        assert not ({CR_, CA_, CO_} & _used), ("gdn_out CB ids clash with the GDN role's", sorted({CR_, CA_, CO_} & _used))
        go_inc = " ".join(f'noc_semaphore_inc(get_noc_addr({x}u, {y}u, {GO["gdna"]}u), 1);' for x, y in GO["gdn_xy"])
        GO_LOCAL = int(os.environ.get("E1_GO_LOCAL", "2"))   # 0: A from the DRAM out; 1: local A (+ DRAM out); 2: local A only
        go_local = ""
        if GO_LOCAL:   # every core multicasts its out tile(s) into all 24 GDN cores' A CB (uniform address), then gdna += 1
            gr_ = GO["rects"]
            mc_ = "\n".join(
                f'  if (my_x[0] >= {r[0]}u && my_x[0] <= {r[2]}u && my_y[0] >= {r[1]}u && my_y[0] <= {r[3]}u) '
                f'noc_async_write_multicast_loopback_src(src, get_noc_multicast_addr({r[0]}u, {r[1]}u, {r[2]}u, {r[3]}u, dst), n * 2048u, {r[4]}u, false);\n'
                f'  else noc_async_write_multicast(src, get_noc_multicast_addr({r[0]}u, {r[1]}u, {r[2]}u, {r[3]}u, dst), n * 2048u, {r[4]}u, false);'
                for r in (gr_[0:5], gr_[5:10]) if r[4])
            go_local = (f'void gdn_out_local(uint32_t src, uint32_t t0, uint32_t n) {{\n'
                        f'  DeviceZoneScopedN("GO-LOC");\n'
                        f'  const uint32_t dst = get_write_ptr({CA_}) + t0 * 2048u;\n{mc_}\n'
                        f'  noc_async_write_barrier();\n  {go_inc}\n}}\n')
            go_inc = ""   # gdn_out_signal() no longer raises gdna
        GO_RD = GO_LOCAL and os.environ.get("E1_GO_RD", "1") == "1"
        go_rd = ""
        if GO_RD:
            # E1_GO_RD (spec v1.91): the handoff multicast + barrier + gdna incs run on the role's READER (NCRISC, NOC1;
            # idle by then) instead of the writer on the congested NOC0: the writer hook posts (src, t0, n) in arg block
            # words and waits for the reader's done word (the out tile is popped right after). XS / XT / XN / XF / XD
            # = payload words after the gdn_out tile list; reset by the writer at program end (the reader's last post is done).
            XB = "{OGO} + 2 + " + str(MAXT)
            mc1 = "\n".join(
                f'    if (my_x[0] >= {r[0]}u && my_x[0] <= {r[2]}u && my_y[0] >= {r[1]}u && my_y[0] <= {r[3]}u) '
                f'noc_async_write_multicast_loopback_src(src, get_noc_multicast_addr({r[2]}u, {r[3]}u, {r[0]}u, {r[1]}u, dst), n * 2048u, {r[4]}u, false);\n'
                f'    else noc_async_write_multicast(src, get_noc_multicast_addr({r[2]}u, {r[3]}u, {r[0]}u, {r[1]}u, dst), n * 2048u, {r[4]}u, false);'
                for r in (gr_[0:5], gr_[5:10]) if r[4])
            inc1 = " ".join(f'noc_semaphore_inc(get_noc_addr({x}u, {y}u, {GO["gdna"]}u), 1);' for x, y in GO["gdn_xy"])
            go_local = (f'void gdn_out_local(uint32_t src, uint32_t t0, uint32_t n) {{\n'
                        f'  DeviceZoneScopedN("GO-LOCW");\n'
                        f'  g_ab[{XB} + 0] = src; g_ab[{XB} + 1] = t0; g_ab[{XB} + 2] = n; g_ab[{XB} + 3] = g_gdn_l + 1;\n'
                        f'  noc_semaphore_wait_min(g_ab + {XB} + 4, g_gdn_l + 1);\n}}\n')
            go_rd = (f'    {{ DeviceZoneScopedN("GO-RD");\n'
                     f'    noc_semaphore_wait_min(g_ab + {XB} + 3, L + 1);\n'
                     f'    const uint32_t src = g_ab[{XB} + 0], n = g_ab[{XB} + 2];\n'
                     f'    const uint32_t dst = get_write_ptr({CA_}) + g_ab[{XB} + 1] * 2048u;\n{mc1}\n'
                     f'    noc_async_write_barrier();\n    g_ab[{XB} + 4] = L + 1;\n    {inc1}\n    noc_async_atomic_barrier(); }}\n')
        go_r = (f'  {{ DeviceZoneScopedN("GO-RING"); cb_reserve_back({CR_}, {RP}); const auto gw = TensorAccessor(TensorAccessorArgs<{{GW_OFF}}>(), {GO["w_addr"]}u, '
                f'get_tile_size({CR_}));\n'
                f'    wou::ring_fill({CR_}, gw.get_noc_addr(g_ab[{{OGO}} + 1] + (L_) * {GO["w_stride"]}u), 0, {GO["wk"]}u * g_ab[{{OGO}}], 16, 2, 13, []() {{}});\n'
                f'    cb_push_back({CR_}, {RP}); }}\n')
        owx = ", ".join(f"{x}u" for x, _ in GO["owners"]); owy = ", ".join(f"{y}u" for _, y in GO["owners"])
        go_w = (f'  {{   // gdn_out of GDN layer L\n'
                f'    {{ DeviceZoneScopedN("GO-A");\n'
                f'    noc_async_atomic_barrier();\n'
                f'    noc_semaphore_wait_min((volatile tt_l1_ptr uint32_t*){GO["gdna"]}u, {len(cores)}u * (L + 1));\n'
                f'    cb_reserve_back({CA_}, {GO["wk"]});\n'
                + ('' if GO_LOCAL else
                f'    const auto oa = TensorAccessor(TensorAccessorArgs<{{OUT_OFF}}>(), crta[{CRTA0} + {WRITER_TENSORS.index("out")}], 2048);\n'
                f'    for (uint32_t t = 0; t < {GO["wk"]}u; ++t) noc_async_read(oa.get_noc_addr(t), get_write_ptr({CA_}) + t * 2048, 2048);\n'
                f'    noc_async_read_barrier();\n') +
                f'    cb_push_back({CA_}, {GO["wk"]}); }}\n'
                f'    noc_semaphore_wait_min((volatile tt_l1_ptr uint32_t*){GO["xow"]}u, 1);\n'
                f'    const uint32_t xo = *(volatile tt_l1_ptr uint32_t*){GO["xow"]}u;\n'
                f'    static constexpr uint32_t owx[16] = {{{owx}}}, owy[16] = {{{owy}}};\n'
                f'    uint32_t tl[{MAXT}];\n'
                f'    for (uint32_t j = 0; j < {MAXT}u; ++j) tl[j] = g_ab[{{OGO}} + 2 + j];\n'
                f'    const uint32_t nt = g_ab[{{OGO}}];\n'
                f'    {{ DeviceZoneScopedN("GO-MM"); cb_wait_front({CO_}, {MAXT}); }}\n'
                f'    DeviceZoneScopedN("GO-SC");\n'
                f'    const auto dg = TensorAccessor(TensorAccessorArgs<{{DBG_OFF}}>(), {GO["dbg_addr"]}u, 2048);\n'
                f'    for (uint32_t j = 0; j < nt; ++j) noc_async_write(get_read_ptr({CO_}) + j * 2048, dg.get_noc_addr(L * 160 + tl[j]), 2048);\n'
                f'    wou::scatter({CO_}, {MAXT}, nt, tl, owx, owy, xo, {GO["cnt"]}u);   // also retires the debug writes\n'
                f'    {{OSIG}}\n'
                + ('    { DeviceZoneScopedN("GO-OSIGB"); noc_async_atomic_barrier(); }\n' if os.environ.get("E1_GO_OSIGB", "0") == "1" else '') +
                f'    }}\n')   # 4-space close: program.py's AR splice (hosted sender on the corners) cuts the layer body at the
                                # first "\n  }\n" after the loop head -- a 2-space close here ended the body early (mesh hang)
        go_c = f'    {{ DeviceZoneScopedN("C-GOMM"); wou::matmul_nt<{"true" if GO.get("lofi", True) else "false"}>({CA_}, {CR_}, {RP}, {CO_}, {MAXT}, {GO["wk"]}, g_ab[{{OGO}}], 4); }}   // short inits only; half-sync fp32 DST: 4 tiles / pass\n'
    rct, roff = ctas([rmap[t] for t in rtens])
    wct, woff = ctas([wmap[t] for t in WRITER_TENSORS])
    if GO:   # gdn_out accessors appended after the step's (reader: the weight slices; writer: the debug output copy)
        go_r = go_r.replace("{GW_OFF}", str(len(rct)))
        rct = rct + ttnn.TensorAccessorArgs(GO["w"]).get_compile_time_args()
        GO_RDALL = GO_RD and os.environ.get("E1_GO_RDALL", "1") == "1"
        go_wr = ""
        if GO_RDALL:
            # E1_GO_RDALL (default on with E1_GO_RD): the whole gdn_out tail (A push, wait the matmul, debug copy, scatter,
            # out_ready) also runs on the reader (NOC1) right after the handoff: on the writer it waited for writer<>()'s
            # trailing writes (S' / stash; wr_out ~60 us at u8) although A was ready ~25 us earlier
            go_wr = go_w.replace("{DBG_OFF}", str(len(rct)))
            rct = rct + ttnn.TensorAccessorArgs(GO["dbg"]).get_compile_time_args()
            go_w = ""
        else:
            go_w = go_w.replace("{OUT_OFF}", str(woff[WRITER_TENSORS.index("out")])).replace("{DBG_OFF}", str(len(wct)))
            wct = wct + ttnn.TensorAccessorArgs(GO["dbg"]).get_compile_time_args()
        OGO = 29 + NG * (len(READER_TENSORS) + len(WRITER_TENSORS)) + 2   # gdn_out payload words: after the per-layer tensors + OAG / OAG_ME
        go_r, go_w, go_c, go_local, go_wr = (x.replace("{OGO}", str(OGO)) for x in (go_r, go_w, go_c, go_local, go_wr))
    hr, hw, hc = (os.path.join(gen_dir, f"gbind_{k}.h") for k in ("r", "w", "c"))
    open(hr, "w").write(_bind_header(ids, rtens, roff, CRTA0))
    open(hw, "w").write(_bind_header(ids, WRITER_TENSORS, woff, CRTA0))
    open(hc, "w").write(_bind_header(ids, [], []))
    W = g.W // 32
    z_tile0, ab_page = (2 * Nk * Kt * 32 + Nv * Vt * 32) // 32, g.QKVZ // 32
    ctrl_bytes = int(tens["ctrl"].shape[-1]) * 4
    assert ctrl_bytes <= CTRL_B, (ctrl_bytes, CTRL_B)  # the ctrl_r / ctrl_w CBs hold the whole ctrl page
    _audit_cb_transfers(tab, layers, Kt, Vt, T, VB)
    hs = 4294967295
    nR, nW = len(rtens), len(WRITER_TENSORS)
    OW, OC, OLR = R_ARGS, R_ARGS + W_ARGS, R_ARGS + W_ARGS + C_ARGS
    OLW = OLR + NG * nR
    S = {k: f"{int(sems[k])}u" for k in ("in_ready", "attn_kv", "vb0", "vb1", "out_ready")}
    if len(out_target) == 2:   # one target core (e.g. the Q root): +1 per GDN core per layer
        ox, oy = out_target
        osig = f'noc_semaphore_inc(get_noc_addr({ox}u, {oy}u, {S["out_ready"]}), 1);'
    else:   # NoC-0 multicast rects [x0, y0, x1, y1, n] * k: +1 on every engine core (each waits 24 (g + 1) itself)
        osig = " ".join(f'noc_semaphore_inc_multicast(get_noc_multicast_addr({r[0]}u, {r[1]}u, {r[2]}u, {r[3]}u, '
                        f'{S["out_ready"]}), 1, {r[4]}u);' for r in (out_target[i:i + 5] for i in range(0, len(out_target), 5))
                        if r[4])
    # E1_OSIG_AGG (default on): out_ready through ONE aggregator (GDN core 0). Every core += 1 on an argblock word of core 0
    # (same L1 address on every core: OAG), core 0 waits 24 (L + 1) and sends the engine signal once with value 24, then
    # retires it (atomic barrier). 24 cores x k multicast atomics each lagged ~20 us in the NoC behind the writers'
    # next-layer traffic (u8: last out signal -> stall end 18-21 us). Core 0 resets the word at program end (every
    # increment of the pass has arrived once its last wait passed). Same idea as gdn_out a9549c3a166.
    agg = os.environ.get("E1_OSIG_AGG", "1") == "1"
    OAG = OLW + NG * nW          # payload word: the aggregation counter (0 at build)
    OAG_ME = OAG + 1             # payload word: 1 on the aggregator core
    AGX, AGY = (lambda q: (q.x, q.y))(dev.worker_core_from_logical_core(cores[0]))
    if agg:
        NC_ = len(cores)
        if len(out_target) == 2:
            sig_n = f'noc_semaphore_inc(get_noc_addr({out_target[0]}u, {out_target[1]}u, {S["out_ready"]}), {NC_});'
        else:
            sig_n = " ".join(f'noc_semaphore_inc_multicast(get_noc_multicast_addr({r[0]}u, {r[1]}u, {r[2]}u, {r[3]}u, '
                             f'{S["out_ready"]}), {NC_}, {r[4]}u);' for r in (out_target[i:i + 5] for i in range(0, len(out_target), 5)) if r[4])
        osig = (f'noc_semaphore_inc(get_noc_addr({AGX}u, {AGY}u, (uint32_t)(g_ab + {OAG})), 1); '
                f'if (g_ab[{OAG_ME}]) {{ noc_semaphore_wait_min(g_ab + {OAG}, {NC_}u * (g_gdn_l + 1)); {sig_n} noc_async_atomic_barrier(); }}')
    # the out_ready signal issued from the reader (NOC1): loop variable g_l, multicast corners swapped for NOC1
    osig_rd = re.sub(r"get_noc_multicast_addr\((\w+), (\w+), (\w+), (\w+), ", r"get_noc_multicast_addr(\3, \4, \1, \2, ",
                     osig.replace("g_gdn_l", "g_l"))
    # GDN_OUT_RD: the same signal from the reader (layer counter g_l, NOC1: multicast rects start / end swapped)
    def _sw(t):
        return [t[i + k] for i in range(0, len(t), 5) for k in (2, 3, 0, 1, 4)]
    if out_rd and len(out_target) != 2:
        tgt_r = _sw(list(out_target))
        mc = lambda v: " ".join(f'noc_semaphore_inc_multicast(get_noc_multicast_addr({r[0]}u, {r[1]}u, {r[2]}u, {r[3]}u, '
                                f'{S["out_ready"]}), {v}, {r[4]}u);' for r in (tgt_r[i:i + 5] for i in range(0, len(tgt_r), 5)) if r[4])
        osig_r = mc(len(cores)) if agg else mc(1)
        if agg:
            osig_r = (f'noc_semaphore_inc(get_noc_addr({AGX}u, {AGY}u, (uint32_t)(g_ab + {OAG})), 1); '
                      f'if (g_ab[{OAG_ME}]) {{ noc_semaphore_wait_min(g_ab + {OAG}, {len(cores)}u * (g_l + 1)); {osig_r} noc_async_atomic_barrier(); }}')
    else:
        osig_r = osig.replace("g_gdn_l", "g_l")
    AB = f"(volatile tt_l1_ptr uint32_t*)(get_common_arg_val<uint32_t>({AB_KIND_GDN}) + {4 * AB_HDR})"
    CR = "volatile tt_l1_ptr uint32_t* crta = (volatile tt_l1_ptr uint32_t*)get_common_arg_addr(0);"
    tr = os.path.join(gen_dir, "g2_reader.cpp")
    open(tr, "w").write(
        f'{DM_OPT}#include "{hr}"\nvoid kernel_main();\nstatic volatile tt_l1_ptr uint32_t* g_ab;\nstatic uint32_t g_l;\n'
        f'inline uint32_t gdn_rt_B() {{ return g_ab[4]; }}\ninline uint32_t gdn_rt_M() {{ return g_ab[5]; }}\n'
        f'#include "{kd}/dataflow/reader_gdn_spec_step.cpp"\n'
        + (f'#include "{WOU}/wo_unit_dm.hpp"\n' if GO else '') +
        f'void gdn_in_wait() {{ noc_semaphore_wait_min((volatile tt_l1_ptr uint32_t*){S["in_ready"]}, g_l + 1); }}\n'
        + (f'void gdn_out_signal() {{ {osig_r} }}\n' if out_rd else '') +
        f'void kernel_main() {{\n  g_ab = {AB};\n  {CR}\n'
        + ('  gdn_cut_reset();\n' if cut_fast else '') +
        f'  for (uint32_t L = 0; L < {NG}; ++L) {{\n    g_l = L;\n'
        f'    for (uint32_t t = 0; t < {nR}; ++t) crta[{CRTA0} + t] = g_ab[{OLR} + L * {nR} + t];\n'
        f'    reader<{Kt}, {Vt}, {Nk}, {Nv}, {T}, 1, 1, {K}, {Lw}, {Ct}, {z_tile0}, {ab_page}, {W}, '
        f'1, 1, {bits(l2_eps)}u, {bits(norm_eps)}u, {ctrl_bytes}, {hs}u, 0, {VB}>(g_ab[0], g_ab[1], g_ab[2], g_ab[3]);\n'
        # gdn_out ring of layer L right after the step's reads of layer L (the slot of L - 1 was popped at the end of L - 1):
        # never blocks, and the next layer's state-first prefetch is not held behind it
        + (go_r.replace("(L_)", "L") if GO else '') +
        (go_rd.replace("{OGO}", str(OGO)) if GO and GO_RD else '') +
        (go_wr.replace("{OSIG}", osig_rd) if GO and GO_RDALL else '') +
        f'  }}\n'
        # GDN_OUT_RD: the reader's out signals retire here; the aggregator's word resets once its last wait passed
        + ('  noc_async_atomic_barrier();\n' + (f'  if (g_ab[{OAG_ME}]) g_ab[{OAG}] = 0;\n' if agg else '') if out_rd else '')
        + '}\n')
    tw = os.path.join(gen_dir, "g2_writer.cpp")
    open(tw, "w").write(
        f'{DM_OPT}#include "{hw}"\nvoid kernel_main();\nstatic volatile tt_l1_ptr uint32_t* g_ab;\nstatic uint32_t g_gdn_l = 0;\n'
        f'inline uint32_t gdn_rt_B() {{ return g_ab[{OW + 11}]; }}\ninline uint32_t gdn_rt_M() {{ return g_ab[{OW + 12}]; }}\n'
        f'#include "{kd}/dataflow/writer_gdn_spec_step.cpp"\n'
        + (f'#include "{WOU}/wo_unit_dm.hpp"\n' if GO else '') +
        f'uint32_t gdn_loop_k() {{ return g_gdn_l + 1; }}\n'
        f'uint32_t gdn_xsem_addr(uint32_t i) {{ return i ? {S["vb1"]} : {S["vb0"]}; }}\n'
        f'void gdn_out_signal() {{ {go_inc if GO else osig} }}\n'
        + (go_local if GO else '')
        + (f'void gdn_in_wait() {{ noc_semaphore_wait_min((volatile tt_l1_ptr uint32_t*){S["in_ready"]}, g_gdn_l + 1); }}\n' if early else '')
        + f'void kernel_main() {{\n  g_ab = {AB};\n  {CR}\n'
        f'  for (uint32_t L = 0; L < {NG}; ++L) {{\n    g_gdn_l = L;\n'
        f'    for (uint32_t t = 0; t < {nW}; ++t) crta[{CRTA0} + t] = g_ab[{OLW} + L * {nW} + t];\n'
        + ('' if early else f'    noc_semaphore_wait_min((volatile tt_l1_ptr uint32_t*){S["in_ready"]}, L + 1);\n')
        + f'    writer<{Kt}, {Vt}, {Nk}, {Nv}, {T}, 1, 1, {Lw}, {Ct}, {Nv * Vt}, {ctrl_bytes}, {hs}u, {VB}>('
        + ", ".join(f"g_ab[{OW + i}]" for i in range(11)) + ");\n"
        + (go_w.replace("{OSIG}", osig) if GO else '') + "  }\n"
        f'  noc_async_atomic_barrier();\n'
        f'  // v1.33: the waiter drains and resets its device-local semaphores at program end (every target reached);\n'
        f'  // attn_kv: the engine raises it twice per attention layer (start, SDPA end), {attn_n} attention layers\n'
        + (f'  noc_semaphore_wait_min((volatile tt_l1_ptr uint32_t*){S["attn_kv"]}, {2 * attn_n});\n' if attn_n else '')
        + "".join(f'  *(volatile tt_l1_ptr uint32_t*){S[k]} = 0;\n' for k in ("in_ready", "attn_kv", "vb0", "vb1"))
        + (f'  *(volatile tt_l1_ptr uint32_t*){GO["gdna"]}u = 0;\n  *(volatile tt_l1_ptr uint32_t*){GO["xow"]}u = 0;\n' if GO else '')
        + (f'  g_ab[{OGO} + 2 + {GO["maxt"]} + 3] = 0; g_ab[{OGO} + 2 + {GO["maxt"]} + 4] = 0;\n' if GO and GO_RD else '')
        + (f'  if (g_ab[{OAG_ME}]) g_ab[{OAG}] = 0;\n' if agg and not out_rd else '')
        + '}\n')
    tc = os.path.join(gen_dir, "g2_compute.cpp")
    open(tc, "w").write(
        f'#include "{hc}"\nvoid kernel_main();\nuint32_t gdn_rt_B();\n'
        f'#include "{kd}/compute/gdn_spec_step.cpp"\n'
        + (f'#include "{WOU}/wo_unit_c.hpp"\n' if GO else '') +
        f'static volatile tt_l1_ptr uint32_t* g_ab;\nuint32_t gdn_rt_B() {{ return g_ab[{OC + 3}]; }}\n'
        f'void kernel_main() {{\n  g_ab = {AB};\n'
        + ('  g_sp_pending = 0;\n' if sp_late else '')
        + f'  for (uint32_t L = 0; L < {NG}; ++L) {{ compute<{Kt}, {Vt}, {T}, {K}, 0, {bits(scale)}u, {bits(1.0 / (Vt * 32))}u, {VB}>('
        f'g_ab[{OC}], g_ab[{OC + 1}], g_ab[{OC + 2}]);\n' + (go_c if GO else '') + '  }\n'
        + ('  gdn_sp_drain();\n' if sp_late else '') + '}\n')
    pre_w = Nv * pre_s
    common = [("GDN_RT_BM", "1"), ("GDN_BATCH", "1")]
    d_r = [("GDN_FAST", "1"), ("GDN_GATE_SCALARS", "1"), ("GDN_EARLY_SRC", "1"), ("GDN_LAZY", "1"),
           ("GDN_PREFIXED", "1"), ("GDN_PRE_W", str(pre_w)), ("GDN_PRE_S", str(pre_s)), ("GDN_STATE_FIRST", "1"),
           ("GDN_IN_WAIT", "1"), ("GDN_ST_RING", str(st_ring))] + common + ([("GDN_NO_EPS_L2", "1")] if lean else []) + (
        [("GDN_CUT_FAST", "1")] if cut_fast else []) + (
        [("GDN_SP_RD", "1")] if sp_rd else []) + ([("GDN_OUT_RD", "1")] if out_rd else [])
    d_w = [("GDN_LAZY", "1"), ("GDN_SFPU_GATE", "1"), ("GDN_GATE_SCALARS", "1"), ("GDN_AB_PAGE", str(ab_page)),
           ("GDN_W_TILES", str(W)), ("GDN_DTB_FP32", "1"), ("GDN_NEA_FP32", "1"), ("GDN_NARROW_GR", "1"),
           ("GDN_VSPLIT", "1"), ("GDN_PREFIXED", "1"), ("GDN_PREFIXED_GATES", "1"), ("GDN_PRE_W", str(pre_w)),
           ("GDN_PRE_S", str(pre_s)), ("GDN_LOOP", "1"), ("GDN_EXT_SEM", "1"), ("GDN_OUT_SIGNAL", "1")] + ([("GDN_OUT_LOCAL", str(GO_LOCAL))] if GO and GO_LOCAL else []) + common + (
        [("GDN_EARLY_REPLAY", "1"), ("GDN_W_IN_WAIT", "1")] if early else []) + ([("GDN_SP_LATE", "1")] if sp_late else []) + (
        [("GDN_RCOEF_FAST", "1")] if early and rcoef_fast else []) + ([("GDN_SP_RD", "1")] if sp_rd else []) + (
        [("GDN_OUT_TILE", "1")] if out_tile else []) + ([("GDN_OUT_RD", "1")] if out_rd else [])
    d_c = [("GDN_BATCH", "1"), ("GDN_ST_RING", str(st_ring))] + ([("GDN_SP_WAIT", "1")] if sp_rd and os.environ.get(
        "E1_SP_WAIT", "0") == "1" else []) + ([("GDN_NO_EPS_L2", "1")] if lean else []) + (
        [("GDN_ZS_LATE", "1")] if zs_late else []) + ([("GDN_EARLY_REPLAY", "1")] if early else []) + (
        [("GDN_SP_LATE", "1")] if sp_late else []) + [(k, "1") for k in ("GDN_FAST", "GDN_FP32_STATE", "GDN_GATE_SCALARS", "GDN_LAZY",
                                                    "GDN_PREFIXED", "GDN_PREFIXED_GATES", "GDN_SFPU_GATE",
                                                    "GDN_SFPU_GATE_FACE", "GDN_VSPLIT", "GDN_VS_TAIL")]
    fphys = dev.worker_core_from_logical_core
    payloads = []
    for i, c in enumerate(cores):
        h, s = i // VB, i % VB
        pxy = []
        for p in range(4):
            if p < VB:
                q = fphys(cores[h * VB + p])
                pxy += [q.x, q.y]
            else:
                pxy += [0, 0]
        pl = [0, h, 0, s, B, M, 0, 0] + [0, h, s] + pxy + [B, M, 0, 0] + [0, 0, s, B, 0, 0]
        pl += [lt[rmap[t]].buffer_address() for lt in layers for t in rtens]
        pl += [lt[wmap[t]].buffer_address() for lt in layers for t in WRITER_TENSORS]
        assert len(pl) == OAG, (len(pl), OAG)
        pl += [0, 1 if i == 0 else 0]   # OAG (aggregation counter), OAG_ME (aggregator flag)
        if GO:   # [nt, slice base page, output tiles (rot order) x maxt]
            assert len(pl) == OGO, (len(pl), OGO)
            pl += [GO["nt"][i], GO["page"][i]] + (list(GO["tiles"][i]) + [0] * GO["maxt"])[:GO["maxt"]]
            pl += [0, 0, 0, 0, 0]   # E1_GO_RD handoff words: src, t0, n, posted (L + 1), done (L + 1)
        payloads.append(pl)
    abt = argblock_tensor(dev, cores, payloads)
    ca = lambda tl: [0, 0, 0, abt.buffer_address(), 0] + tl
    utd = [ttnn.UnpackToDestMode.Default] * 64
    for n in UNPACK_TO_DEST:
        utd[ids[n][0]] = ttnn.UnpackToDestMode.UnpackToDestFp32
    ccfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                        math_approx_mode=False, dst_full_sync_en=False)
    ccfg.unpack_to_dest_mode = utd
    ks = [ttnn.KernelDescriptor(kernel_source=tr, core_ranges=crs, compile_time_args=rct, defines=d_r,
                                common_runtime_args=ca([tens[rmap[t]].buffer_address() for t in rtens]),
                                runtime_args=ttnn.RuntimeArgs(), config=ttnn.DataMovementConfigDescriptor(
                                    processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1)),
          ttnn.KernelDescriptor(kernel_source=tw, core_ranges=crs, compile_time_args=wct, defines=d_w,
                                common_runtime_args=ca([tens[wmap[t]].buffer_address() for t in WRITER_TENSORS]),
                                runtime_args=ttnn.RuntimeArgs(), config=ttnn.DataMovementConfigDescriptor(
                                    processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0)),
          ttnn.KernelDescriptor(kernel_source=tc, core_ranges=crs, compile_time_args=[], defines=d_c,
                                common_runtime_args=ca([]), runtime_args=ttnn.RuntimeArgs(), config=ccfg)]
    if GO:   # CB 41 ring (the weight format, BFP4), 42 A (24 bf16 tiles), 43 out slot (maxt bf16 tiles)
        wfmt = GO["w"].dtype
        wpg = ttnn.Tile([32, 32]).get_tile_size(wfmt)
        for idx, dt, n, pg in ((CR_, wfmt, GO["wk"] * GO["maxt"], wpg), (CA_, ttnn.bfloat16, GO["wk"], 2048), (CO_, ttnn.bfloat16, GO["maxt"], 2048)):
            cbs.append(ttnn.CBDescriptor(total_size=n * pg, core_ranges=crs, format_descriptors=[ttnn.CBFormatDescriptor(
                buffer_index=idx, data_format=dt, page_size=pg)]))
    io = [abt] + [x for lt in layers for x in (lt["win_a"], lt["win_b"], lt["ring"], lt["ctrl"], lt["taps"], lt["dtb"],
                                                lt["nea"], lt["weight"], lt["pre"], lt["out"])] + [layers[0]["qkv"]] + ([GO["w"], GO["dbg"]] if GO else [])
    info = dict(cb_ids=sorted({ids[n][0] for n, _, _, _ in tab}), cb_bytes=sum(e * nb for _, e, nb, _ in tab) + (sum(c_.total_size for c_ in cbs[-3:]) if GO else 0),
                ab_words=len(payloads[0]), NG=NG, st_ring=st_ring, early_replay=early, sp_late=sp_late, lean=lean, rcoef_fast=rcoef_fast, sp_rd=sp_rd, zs_late=zs_late, cut_fast=cut_fast, out_tile=out_tile)
    return dict(kernels=ks, cbs=cbs, io=io, info=info)
