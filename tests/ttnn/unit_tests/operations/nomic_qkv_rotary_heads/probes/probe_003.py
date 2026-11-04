import ttnn
from eval.golden_tests.nomic_qkv_rotary_heads.helpers import model_layout, ln_layout
from ttnn.operations.nomic_qkv_rotary_heads import circular_buffer_bytes

dev = ttnn.open_device(device_id=0)
try:
    for s in [(2, 1, 512, 768), (3, 1, 400, 768), (4, 1, 512, 768), (8, 1, 512, 768), (7, 1, 512, 768)]:
        mc = model_layout(s, dev)
        print("SHAPE", s, "cb_bytes", circular_buffer_bytes(s))
        print("  ", mc.memory_layout, mc.buffer_type)
        print("  shard_spec", mc.shard_spec)
        print("  nd", mc.nd_shard_spec)
finally:
    ttnn.close_device(dev)
