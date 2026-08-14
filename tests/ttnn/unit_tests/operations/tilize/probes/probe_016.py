import tests.ttnn.unit_tests.operations.tilize._bench_tilize as B
import torch, ttnn
from ttnn.operations.tilize import validate
from ttnn.operations.tilize import tilize_program_descriptor as pd

device = ttnn.open_device(device_id=0)
_L1 = ttnn.BufferType.L1
_ROW = ttnn.ShardOrientation.ROW_MAJOR


def _crs(n):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(n - 1, 0))})


def H(shape, n):
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, _L1, ttnn.ShardSpec(_crs(n), (shape[-2] // n, shape[-1]), _ROW)
    )


def W(shape, n):
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, _L1, ttnn.ShardSpec(_crs(n), (shape[-2], shape[-1] // n), _ROW)
    )


def plc(shape, inc, outc):
    t = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=inc or ttnn.DRAM_MEMORY_CONFIG,
    )
    plan = validate(t, outc, dtype=ttnn.bfloat16)
    o = ttnn.allocate_tensor_on_device(
        ttnn.Shape(plan.target), plan.out_dtype, ttnn.TILE_LAYOUT, device, plan.out_memory_config
    )
    d = pd.create_program_descriptor(t, o, plan)
    return (d.kernels[0].compile_time_args[1], d.kernels[1].compile_time_args[0], d.kernels[0].core_ranges.num_cores())


S = [1, 1, 1024, 256]
CASES = {
    "dram->H8 (512B read, keep local)": (S, None, H(S, 8)),
    "dram->W8 (64B read, gate off)": (S, None, W(S, 8)),
    "narrowW4->H8 (128B, gate off)": (S, W(S, 4), H(S, 8)),
    "fullH8->H2 (512B, keep local)": (S, H(S, 8), H(S, 2)),
    "H8->dram (source-local, never gated)": (S, H(S, 8), None),
    "small dram->H4 128B (gate off)": ([1, 1, 512, 64], None, H([1, 1, 512, 64], 4)),
    "same-spec H8 (both local)": (S, H(S, 8), H(S, 8)),
}
for name, (sh, inc, outc) in CASES.items():
    print(f"{name}: placements+cores={plc(sh,inc,outc)}")
# measured effect of the gate on the two shapes that lost
for name, sh, inc, outc in [
    ("dram->W8", S, None, W(S, 8)),
    ("narrowW4->H8", S, W(S, 4), H(S, 8)),
    ("dram->H4 small", [1, 1, 512, 64], None, H([1, 1, 512, 64], 4)),
    ("dram->H8 keeps its win", S, None, H(S, 8)),
]:
    on = B._measure(device, sh, ttnn.bfloat16, in_mem_config=inc, out_mem_config=outc, label=f"{name}/gate=1")
    off = B._measure(
        device,
        sh,
        ttnn.bfloat16,
        in_mem_config=inc,
        out_mem_config=outc,
        levers=dict(xfer_gate=0),
        label=f"{name}/gate=0",
    )
    print(f"GATE {name}: gate_on={on} gate_off={off} speedup={off/on:.2f}x")
ttnn.close_device(device)
