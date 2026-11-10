"""Megakernel ↔ qwen36 verify integration (DESIGN_integration.md). Flag QWEN36_R16_MEGAKERNEL:
  unset / "0" : off (today's op-by-op verify, unchanged)
  "stage0"    : plumbing probe. The verify trace also launches a one-core-per-device generic_op that reads this
                bucket's phase table and writes a checksum. The layer loop still runs op-by-op, so outputs are
                bit-identical to off.
  "stage1"    : Stage1Verify runs the engine of QWEN36_R16_MK_ENGINE_KIND (default identity) + the stage0 probe.
  "1"         : release mode: Stage1Verify with the real engine (kind mk by default), no probe. Falls back op-by-op with
                a logged reason while engine/program.py is missing.
"""

import os

from models.demos.blackhole.qwen36.tt.megakernel.integ.phase_table import WORDS, build_phase_table, checksum, summary

_KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")  # absolute: the tree may be an overlay


def megakernel_mode():
    v = os.environ.get("QWEN36_R16_MEGAKERNEL", "0").strip().lower()
    return None if v in ("", "0", "off", "false") else v


class MegakernelVerify:
    """Per-model megakernel state: per-bucket phase tables (persistent device tensors) and probe outputs.

    Built once at verify setup (``model._vfy_mk = MegakernelVerify(model)``), when ``megakernel_mode()`` is set.
    ``pre_layers(n_users)`` is called inside the traced verify body, right before the layer loop."""

    def __init__(self, model):
        import ttnn

        self.ttnn = ttnn
        self.mode = megakernel_mode()
        self.mesh = model.mesh_device
        self.fa = [bool(l.is_full_attention) for l in model.layers]
        self.tables = {}  # n_users -> (records, device table tensor)
        self.probe_out = {}  # n_users -> device uint32 [1,1,1,32] RM (persistent: the trace writes it)
        self.fallback_reason = None
        # launch accounting (stage0 proof that the probe runs inside every replay): eager body calls outside a trace
        # capture + execute_trace replays, per bucket; the device counter (out[4]) must equal their sum.
        self.capturing = False
        self.eager = {}
        self.replays = {}
        if self.mode not in ("stage0", "stage1", "1"):
            self.fallback_reason = f"QWEN36_R16_MEGAKERNEL={self.mode}: unknown mode; layer loop op-by-op"

    # ---- tables -------------------------------------------------------------------------------------------------
    def _table(self, n_users):
        if n_users in self.tables:
            return self.tables[n_users]
        import torch

        ttnn = self.ttnn
        recs = build_phase_table(self.fa, n_users)
        host = torch.tensor(recs, dtype=torch.int64).to(torch.int32).reshape(1, 1, len(recs), WORDS)
        rep = ttnn.ReplicateTensorToMesh(self.mesh)
        dev = ttnn.from_torch(host, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh,
                              memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep)
        out = ttnn.from_torch(torch.zeros(1, 1, 1, 32, dtype=torch.int32), dtype=ttnn.uint32,
                              layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                              mesh_mapper=rep)
        self.tables[n_users] = (recs, dev)
        self.probe_out[n_users] = out
        return self.tables[n_users]

    def prepare(self, buckets):
        """Allocate every bucket's table + probe output BEFORE trace capture (no allocations inside a capture)."""
        for b in buckets:
            self._table(int(b))
        return {b: summary(self.tables[int(b)][0]) for b in buckets}

    # ---- stage0 probe -------------------------------------------------------------------------------------------
    def _probe_program(self, tbl, out, n_phases):
        ttnn = self.ttnn
        core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
        cta = ttnn.TensorAccessorArgs(tbl).get_compile_time_args() + ttnn.TensorAccessorArgs(out).get_compile_time_args()
        scratch_cb = ttnn.CBDescriptor(
            total_size=4096, core_ranges=core,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=24, data_format=ttnn.uint32, page_size=4096)])
        # the kernel takes its L1 scratch from CB 24 (get_write_ptr), so no L1 address is passed
        rt = ttnn.RuntimeArgs()
        rt[0][0] = [tbl.buffer_address(), out.buffer_address(), n_phases]
        k = ttnn.KernelDescriptor(
            kernel_source=f"{_KDIR}/mk_probe.cpp", core_ranges=core, compile_time_args=cta, runtime_args=rt,
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0))
        return ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[scratch_cb])

    def pre_layers(self, n_users):
        """Called inside the traced verify body. stage0: launch the probe (does not touch the residual)."""
        if self.mode not in ("stage0", "stage1"):  # stage1 keeps the probe: proves the bucket trace replays
            return
        recs, tbl = self._table(n_users)
        out = self.probe_out[n_users]
        if not self.capturing:
            self.eager[n_users] = self.eager.get(n_users, 0) + 1
        self.ttnn.generic_op([tbl, out], self._probe_program(tbl, out, len(recs)))

    def count_replay(self, n_users):
        """Called by model.verify_traced right after execute_trace of a verify trace for ``n_users`` rows."""
        self.replays[n_users] = self.replays.get(n_users, 0) + 1

    def report(self):
        """Per bucket: device launch counters (every device), expected = eager + replays, checksum ok on all devices.
        Synchronizes the mesh (call after the run, outside any trace)."""
        import torch

        ttnn = self.ttnn
        ttnn.synchronize_device(self.mesh)
        rep = {}
        for b, (recs, _) in sorted(self.tables.items()):
            want = checksum(recs)
            got = ttnn.to_torch(self.probe_out[b], mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=0))
            rows = (got.reshape(-1, 32).to(torch.int64) & 0xFFFFFFFF).tolist()
            exp = self.eager.get(b, 0) + self.replays.get(b, 0)
            launched = [r[4] for r in rows]
            ok_sum = all((r[0], r[1], r[2]) == want and r[3] == 0x4D4B5030 for r in rows) if exp else None
            rep[b] = dict(expected=exp, eager=self.eager.get(b, 0), replays=self.replays.get(b, 0),
                          device_counts=sorted(set(launched)), checksum_ok=ok_sum,
                          ok=(exp == 0 and set(launched) == {0}) or (ok_sum and set(launched) == {exp}))
        return rep

    def check_probe(self, n_users):
        """Host check after a replay: every device's probe output equals the host checksum of the bucket's table."""
        import torch

        ttnn = self.ttnn
        recs, _ = self.tables[n_users]
        want = checksum(recs)
        got = ttnn.to_torch(self.probe_out[n_users], mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=0))
        rows = got.reshape(-1, 32).to(torch.int64) & 0xFFFFFFFF
        bad = [i for i, r in enumerate(rows.tolist()) if (r[0], r[1], r[2]) != want or r[3] != 0x4D4B5030]
        return want, rows.shape[0], bad
