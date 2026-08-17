"""No-device control importing the pre-existing shared Qwen36 dependency."""

from models.demos.blackhole.qwen36.tt.layer import Qwen36DecoderLayer

print(f"SHARED_QWEN_IMPORT_CONTROL_OK class={Qwen36DecoderLayer.__name__}")
