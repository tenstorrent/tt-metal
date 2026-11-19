# Resident admission repair diff

Diff against round1 implementation; blank context lines normalized for display.

```diff
diff --git a/models/autoports/ornith_ai_ornith_1_5_9b/tt/generator.py b/models/autoports/ornith_ai_ornith_1_5_9b/tt/generator.py
index 25289e6a..dd05ba98 100644
--- a/models/autoports/ornith_ai_ornith_1_5_9b/tt/generator.py
+++ b/models/autoports/ornith_ai_ornith_1_5_9b/tt/generator.py
@@ -818,6 +818,11 @@ class OrnithGenerator(Generator):
                 if position >= self.max_seq_len or needed > tables.shape[1] or torch.any(tables[slot, :needed] < 0):
                     raise ValueError("Every active causal page through the current position must be mapped")
             tables = tables.to(torch.int32)
+        else:
+            # A remap permutes both positions and page rows, so admission of all
+            # current slots is equivalent before that transition. Validate here,
+            # before remap writes or trace submission can mutate device state.
+            self._validate_resident_window(previous, 1)
         if commands.reload_sampling_params:
             self._validate_sampling(sampling_params)
         histories = (None, None)
@@ -970,6 +975,20 @@ class OrnithGenerator(Generator):
             if state.cache is cache:
                 state.resident_ready = False

+    def _validate_resident_window(self, state, steps):
+        """Admit the entire window using CPU metadata, without device transfers."""
+        import torch
+
+        for slot, position in enumerate(state.host_positions):
+            if position < 0:
+                continue
+            last = position + state.position_advances + steps - 1
+            if last >= self.max_seq_len:
+                raise ValueError("Resident decode exceeds configured context")
+            needed = last // 32 + 1
+            if needed > state.host_tables.shape[1] or torch.any(state.host_tables[slot, :needed] < 0):
+                raise ValueError("Every active causal page through the resident decode window must be mapped")
+
     def decode_resident(self, steps, *, kv_cache=None):
         """Queue fixed-step sampled decode without host transfers or synchronization.

@@ -990,11 +1009,7 @@ class OrnithGenerator(Generator):
             raise ValueError("Initialize resident decode inputs before queuing steps")
         if any(h.completion is None for h, _, _ in self.submissions.values()):
             raise ValueError("Drain outstanding submissions before resident decode")
-        if any(
-            position >= 0 and position + state.position_advances + steps > self.max_seq_len
-            for position in state.host_positions
-        ):
-            raise ValueError("Resident decode exceeds configured context")
+        self._validate_resident_window(state, steps)
         for _ in range(steps):
             self._execute(state, host_sampling=False)
         return state.tokens
```
