# SPDX-License-Identifier: Apache-2.0
"""Isolate the W2 prepare/quantize 'Tensor is not allocated' failure."""
import os, sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
os.chdir(THIS)
sys.path.insert(0, str(THIS))
import ttnn, utils  # noqa

dev = utils.DeviceGetter.get_device((4, 8))
CE = THIS / "moe_io" / "ce_cache"
DRAM = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
EPD, HIDDEN, N = 8, 7168, 2048


def st(name, t):
    try:
        print(
            f"  {name}: shape={tuple(t.shape)} dtype={t.dtype} layout={t.layout} "
            f"storage={t.storage_type()} on_device={t.is_allocated() if hasattr(t,'is_allocated') else '?'}"
        )
    except Exception as e:
        print(f"  {name}: introspect err {e}")


w2_raw = ttnn.load_tensor(str(CE / "main_const_eval_39.tensorbin"), device=dev)
st("w2_raw", w2_raw)
w2 = ttnn.to_layout(
    ttnn.typecast(w2_raw, ttnn.DataType.BFLOAT16, memory_config=DRAM), ttnn.Layout.ROW_MAJOR, None, memory_config=DRAM
)
st("w2(bf16,RM)", w2)

print("calling prepare_w2_tensor_for_moe_compute(E=8)...")
w2_prepped = ttnn.experimental.prepare_w2_tensor_for_moe_compute(w2, L=1, E=EPD, N=N, K=HIDDEN)
st("w2_prepped", w2_prepped)
print("  is on device?", w2_prepped.device() if hasattr(w2_prepped, "device") else "?")

mc = ttnn.experimental.get_weight_mem_configs(
    dev, num_layers=1, experts_per_device=EPD, hidden_size=HIDDEN, intermediate_size=N, has_bias=False
)
print("w2 mem_config:", mc.w2)

# try a plain from_device to see if THAT is the failure
try:
    h = ttnn.from_device(w2_prepped)
    print("from_device OK; host shape", tuple(h.shape))
except Exception as e:
    print("from_device FAILED:", repr(e))

try:
    tt_w2 = ttnn.experimental.quantize_weights_via_host(w2_prepped, dtype=ttnn.bfloat4_b, memory_config=mc.w2)
    st("tt_w2", tt_w2)
    print("QUANTIZE OK")
except Exception as e:
    print("QUANTIZE FAILED:", repr(e))

if utils.DeviceGetter._instance is not None:
    ttnn.close_mesh_device(utils.DeviceGetter._instance)
    utils.DeviceGetter._instance = None
