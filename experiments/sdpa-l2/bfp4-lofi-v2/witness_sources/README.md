# Historical producer-assertion witnesses

These thirteen `.source` files are byte-exact snapshots of producer versions
already pinned by existing checkpoint records. Their filenames are SHA-256
digests of their complete original bytes. They are deliberately not `.py`
files, so ordinary Python/C++ source-formatting passes do not rewrite them.
Do not format or regenerate them from subsequently changed producers.

The directory's `.source` extension avoids Python/C++ formatters, but does not
exempt text from generic end-of-file/whitespace fixers. Some historical bytes
intentionally retain extra final newlines. The local `.gitattributes` setting
disables Git line-ending normalization and suppresses its blank-at-EOF warning;
it is not a pre-commit exemption.
Do not run mutating fixers over these witnesses without an explicit archival
or exclusion plan. No repository-wide hook configuration was changed for this
research checkpoint, and a full pre-commit PASS is not claimed.

| Original producer | Snapshot SHA-256 | Why needed |
|---|---|---|
| `identity4_streaming.py` | `4e54b83d0f45e0b2b155d2f1783118987b5a1065284f0e8dba8902af05bfa7cf` | Fixed K512 builder assignment omitted from case metadata |
| `value_centered_fullchip.py` | `f3bdbde5b553a30322fd06c9c1e7779fcb93974ef44e82de45a188d75bdc611d` | Finite-output and during-run source-stability assertions |
| `value_centered_b8_fullchip.py` | `f33b58b1cf7f2ef27f906a682932e68407a1bb7edc8b82ee023a247efd1c360d` | Finite-output and during-run source-stability assertions |
| `adaptive_fullchip.py` | `285020c0039558686c458439a172c33c56c6ba1882ecbf4fa3936a50c3fefdf7` | Finite-output guard and source-stability assertion |
| `exp_lut_macro_streaming.py` | `551ff73616819c527c95ccb5b95a303b08850d6e9720171b0f4f3f79ed2784eb` | Finite-output and source-stability assertions |
| `exp_lut_macro_resident.py` | `5b64ea58f3ea20aca03d90f9350fa17ca2edbfabb174696f50dc64fca57296f4` | Finite-output and source-stability assertions |
| `Vtransposed_fullchip.py` | `25ff5d7a745d18259196cd5fee50f41b9b2aab455707122074189a61152a278d` | Original host hashes and device-input bitwise immutability assertions |
| `fullchip.py` | `b80ec4fa1a4db90384b4dfac5c7d4a614c28fd3c79494560411b73987c913038` | Native-storage enabled exact-preprocessing assertion and K512 assignment |
| `hifi2_native_fullchip.py` | `ead90b029b39f778b5dbe3e0d751c271b4e36fc652aaec4c6358fae75f38fafa` | Selected copied build's exact-preprocessing assertion and K512 assignment |
| `hifi2_lut_fullchip.py` | `8d029715ab78b20ffe96b841eb0750b5a212b4fdc839ab3e3a7b35b4a1f2c5c3` | Selected copied build's exact-preprocessing assertion and K512 assignment |
| `hifi2_bf16_lut_fullchip.py` | `5297432e062bf6800fee1edb77dfe4d12533bde999d0ff785799c1f02877f818` | Fixed K512 assignment; this producer records exact-preparation counts explicitly |
| `combined_recipe_fullchip.py` | `77d71ac3998f8e6f3f205190f1f03b547b8d1e9045fa3c3873bcff0a5cb39dc2` | Original host hashes and device-input bitwise immutability assertions |
| `paired_vaxis_timing.py` | `1838e8c04b180feacb75f86ea80c0d99b8a919374926ad7f4ef63e766e057394` | Original-input FP64 reference assignment; paired records otherwise have explicit gates |

This is the minimal set of unique producer/hash pairs used by the validator's
current assertion-witness calls, not a complete source archive. There is one
recorded version per producer. All snapshots were captured while their source
bytes still matched those recorded hashes, before the planned formatting pass.

The read-only validator first tries the current producer. If its hash differs
from the record or it is unavailable, it looks up the snapshot by the record's
hash. It checks the snapshot's full SHA-256 **before** AST parsing. A missing
snapshot leaves the gate PENDING; wrong bytes at the requested hash fail the
gate. Snapshot code is never executed or imported. The audit records which
source supplied each witness and separately reports current-source drift and
historical manifest omissions. Existing result manifests are not backfilled.

An assertion witness only establishes the corresponding guard/assignment in
the pinned producer source. It is not a new raw-tensor check, device rerun,
universal numerical guarantee, or independent proof of historical execution.
