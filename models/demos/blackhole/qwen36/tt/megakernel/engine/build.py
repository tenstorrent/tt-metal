"""Megakernel engine builder (integration track, spec v0.9.3 §integration contract).

    build_engine(mesh, layer_types, n_users, weights_mk, kv_desc, gdn_desc, fabric_ctx, T=4) -> EngineProgram

EngineProgram (one generic_op per verify pass, built once per bucket before trace capture):
    .R               rows per activation transfer: 4*n_users rounded up to {4, 8, 16, 32}
    .replaces_layers True: the program IS the 64-layer stack. False: stand-in; the op-by-op layer loop still runs on x_out
    .alloc_x_out()   persistent replicated DRAM residual out [1, 1, n_users*T, 5120] bf16 TILE (called in prepare, not in
                     a capture)
    .launch(x_in, x_out, per_replay)  inside the trace: x_in replicated DRAM un-normed residual -> x_out. per_replay holds
                     the bucket's in-body tensors (addresses only go into runtime args)

Host guarantees (contract item 1): a 1x8 FABRIC_1D_RING on axis 1 with 8 KB packets at device open, and no other op on
those links during the generic_op. The engine opens/closes its own fabric connections and resets its own monotonic
global semaphores at program end; the host never resets them.

Kinds (QWEN36_R16_MK_ENGINE_KIND):
    identity (default until M3): one core per device copies x_in -> x_out (kernels/mk_ident_copy.cpp). Proves the stage1
             plumbing and the trace; outputs stay bit-identical because the layer loop still runs.
    mixed3   the real engine (M3: mixed3 attention + v12 MLP + MLP-AR). Not assembled yet: raises with the reason, so
             Stage1Verify falls back to op-by-op.
"""

import os

DIM = 5120
_KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")


def engine_kind():
    return os.environ.get("QWEN36_R16_MK_ENGINE_KIND", "identity").strip().lower()


def rows_R(n_users):
    r = 4 * int(n_users)
    for c in (4, 8, 16, 32):
        if r <= c:
            return c
    raise ValueError(f"n_users={n_users}: R={r} > 32 (one tile of rows)")


class EngineProgram:
    def __init__(self, mesh, n_users, T, kind):
        self.mesh, self.n_users, self.T, self.kind = mesh, int(n_users), int(T), kind
        self.R = rows_R(n_users)
        self.rows = self.n_users * self.T
        self.replaces_layers = kind != "identity"
        self.launches = 0  # host-side count of launch() calls (eager + captures)

    def alloc_x_out(self):
        import torch
        import ttnn

        return ttnn.from_torch(
            torch.zeros(1, 1, self.rows, DIM, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=self.mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh))

    def launch(self, x_in, x_out, per_replay):
        raise NotImplementedError


class IdentityEngine(EngineProgram):
    CORES = 8  # copy split over a row of cores; page = one bf16 tile (2 KB)
    BATCH = 8

    def launch(self, x_in, x_out, per_replay):
        import ttnn

        assert x_in.dtype == ttnn.bfloat16 and x_in.layout == ttnn.TILE_LAYOUT, (x_in.dtype, x_in.layout)
        assert tuple(x_in.shape) == tuple(x_out.shape) == (1, 1, self.rows, DIM), (x_in.shape, x_out.shape)
        page = 32 * 32 * 2
        n_pages = -(-self.rows // 32) * (DIM // 32)
        per = -(-n_pages // self.CORES)
        rng = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.CORES - 1, 0))])
        cta = ttnn.TensorAccessorArgs(x_in).get_compile_time_args() + ttnn.TensorAccessorArgs(x_out).get_compile_time_args()
        rt = ttnn.RuntimeArgs()
        for c in range(self.CORES):
            p0 = min(c * per, n_pages)
            rt[c][0] = [x_in.buffer_address(), x_out.buffer_address(), p0, max(0, min(per, n_pages - p0)), page, self.BATCH]
        cb = ttnn.CBDescriptor(
            total_size=page * self.BATCH, core_ranges=rng,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=24, data_format=ttnn.bfloat16, page_size=page)])
        k = ttnn.KernelDescriptor(
            kernel_source=f"{_KDIR}/mk_ident_copy.cpp", core_ranges=rng, compile_time_args=cta, runtime_args=rt,
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0))
        self.launches += 1
        ttnn.generic_op([x_in, x_out], ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[cb]))


def build_engine(mesh, layer_types, n_users, weights_mk, kv_desc, gdn_desc, fabric_ctx, T=4):
    kind = engine_kind()
    if kind == "identity":
        return IdentityEngine(mesh, n_users, T, kind)
    if kind == "mixed3":
        raise NotImplementedError(
            "mixed3 engine not assembled yet (needs engine/layout_mlp.py, layout_attn.py, ar_phase.py; milestone M3)")
    raise ValueError(f"QWEN36_R16_MK_ENGINE_KIND={kind}: unknown")
