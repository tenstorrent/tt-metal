# Hooks the DFlash spec path needs in shared files (PROPOSAL, no edits made; decode agent owns tt/decode.py, tt/layer.py decode methods)

1. Capture of the 5 target hidden states (outputs of layers 0,11,23,35,47, layer 47 PRE final norm; DFLASH_NOTES convention) for every token row of a decode/verify step:
   * `tt/decode.py` `Decoder._forward`: inside `for lyr in m.layers:` keep `h2` after layers in CAPTURE = {0,11,23,35,47} (a list of device tensors [1,1,32,4096], replicated over columns),
     after the loop `feat = ttnn.concat(caps, dim=3)` -> [1,1,32,20480], returned as a 4th persistent trace output (`vals, idx, lg, feat`); `Decoder.run` keeps `self.feat` on device (no host readback).
     No change to layer.py: the layer outputs are already the returned `h2`.
   * The drafter consumes `feat` on device: its fc is N-sharded over columns (each column: full K=20480 x 512 outputs) + all_gather over columns for hidden_norm when M is small (decode: 32 rows/row);
     the current tt/dflash.py K-shards fc and all-reduces (right for prefill-size M); both are in the module, select by M.
2. Prefill capture (drafter context for the prompt): in `tt/prefill.py` (mine) the chunk forward also returns the same 5 tensors per chunk ([1,1,CH,20480]) feeding `DFlashDrafter.append_ctx`.
3. Drafter KV: per user per layer ring [1024,128] K and V per column (20 KB per position over all 5 layers and 8 columns), written with the same paged_update/fill ops the target uses (contiguous ring, no paging needed).
4. Verify: see spec_verify.md (virtual users: batch 8U in the decode path; decode.py needs a `block` parameter that expands positions to pos+t and repeats page-table rows).
Files that would be touched: tt/decode.py (Decoder._forward, Decoder.run), tt/prefill.py (chunk forward), new tt/dflash.py, tt/spec.py. No change to tt/attention.py decode methods until the verify design is agreed.
