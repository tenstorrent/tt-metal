# Page-table shadow recovery

After a same-cache device-page-table rebind invalidates the CPU shadow, the serving adapter accepts a complete host table covering every bound row once. It reconstructs zero-padded rows without device readback and rejects partial updates that cannot preserve unknown rows. The known-shadow path is unchanged.

Validation: four host repair tests and 21 B2 harness tests passed. The reduced actual serial arm completed post-rebind decode with state/KV preservation and released traces. Independent review accepts this repair only. The complete B2 run failed a separate inactive-token comparison; that failure is preserved and remains under investigation. No build required for this Python-only change. No performance or Stage11 acceptance is claimed.
