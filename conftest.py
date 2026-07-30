--- a/models/tt_transformers/model_config.py
+++ b/models/tt_transformers/model_config.py
@@ -123,6 +123,10 @@
     "gemma3": {
         "rms_norm_add_unit_offset": True,
         "embed_scale": True,
         "sliding_window": True,
+    "gemma-2-2b-it": {
+        "rms_norm_add_unit_offset": True,
+        "embed_scale": True,
+        "sliding_window": True,
+        "sliding_window_pattern": "alternating",
     },
     "medgemma-27b": {
         "rms_norm_add_unit_offset": True,
         "embed_scale": True,
