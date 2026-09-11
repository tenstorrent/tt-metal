# Repro: tiled reshape_view with a row width that is not a multiple of 8 bf16 elements returns zeros for every
# 16B-misaligned segment on Quasar (craq-sim aborts; RTL emulator returns wrong data silently).
# Root cause: data_movement/common/kernels/common.hpp copy_via_memmove() adds MEM_L1_UNCACHED_BASE to a source
# pointer that DataflowBuffer::get_read_ptr() already returns as the uncached alias on Quasar DM (double offset).
# usage: python tests/scripts/quasar_reshape_misaligned_repro.py [aligned]   (default = misaligned case)
import sys
import torch
import ttnn

aligned = len(sys.argv) > 1 and sys.argv[1] == "aligned"
in_shape, out_shape = ((1, 1, 32, 64), (1, 1, 64, 32)) if aligned else ((1, 1, 32, 36), (1, 1, 36, 32))
torch.manual_seed(0)
dev = ttnn.open_device(device_id=0)
t = torch.randn(*in_shape, dtype=torch.bfloat16)
# host tilize + to_device (from_torch(device=...) would run the legacy on-device tilize, which is not supported on Quasar)
x = ttnn.to_device(
    ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
)
y = ttnn.reshape(x, out_shape)
got = ttnn.to_torch(y)
ref = t.reshape(out_shape)
mism = (got != ref).sum().item()
zeros = (got == 0).sum().item()
print(
    f"RESHAPE_REPRO {'aligned' if aligned else 'misaligned'} {in_shape}->{out_shape} mismatches={mism}/{ref.numel()} zeros_in_output={zeros} "
    f"{'PASS' if mism == 0 else 'FAIL'}",
    flush=True,
)
ttnn.close_device(dev)
sys.exit(0 if mism == 0 else 1)
