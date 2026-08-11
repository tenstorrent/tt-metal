"""Throwaway: how many GCBs does a real multi-layer model build allocate?"""

import os

import ttnn
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

from models.experimental.deepseek_v4_flash.tests.test_full_model_decode_demo import _DEFAULT_MODEL_DIR
from models.experimental.deepseek_v4_flash.tt.model import DeepSeekV4Model
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader

created = []
_real = ttnn.experimental.create_global_circular_buffer_for_tensor_prefetcher


def _counting(device, bank_to_receivers, size, *a, **kw):
    gcb = _real(device, bank_to_receivers, size, *a, **kw)
    created.append((id(device), size, sum(crs.num_cores() for _, crs in bank_to_receivers)))
    return gcb


ttnn.experimental.create_global_circular_buffer_for_tensor_prefetcher = _counting

loader = DeepseekV4WeightLoader(_DEFAULT_MODEL_DIR)
config = DeepseekV4Config.from_pretrained(loader.snapshot_dir)
config._attn_implementation = "eager"

device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
model = DeepSeekV4Model(
    config,
    loader,
    device,
    weight_dtype=ttnn.bfloat4_b,
    max_layers=int(os.environ.get("LAYERS", "4")),
    use_submeshes=True,
    use_prefetcher=True,
)
print(f"\nlayers={model.num_layers}  GCBs created={len(created)}")
per_device = {}
for dev_id, size, receivers in created:
    per_device.setdefault(dev_id, []).append((size, receivers))
for dev_id, gcbs in per_device.items():
    print(f"  device={dev_id}: {len(gcbs)} GCB(s)")
    for size, receivers in gcbs:
        print(f"    size={size} B ({size/1024:.0f} KiB) receivers={receivers}")
ttnn.close_mesh_device(device)
