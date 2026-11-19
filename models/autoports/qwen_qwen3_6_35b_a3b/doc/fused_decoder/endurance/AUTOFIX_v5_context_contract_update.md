# Selected v5 context evidence update

CPU-only metadata reconciliation; no device execution or production/harness edits. `doc/context_contract.json` now binds selected v5 source `8ac733159befba919abf705603f3aada9d105195c7b495d8cd25aa65e531769f` to the fresh `linear_long.json` and `full_long.json` plus their provenance hashes. Both sidecars identify this source, TTNN device execution, fused decoder, physical chip0 and serial execution. The selected snapshot and current production source hash match.

Both layer kinds passed prefill262143 and262144, plus three deterministic traced decodes at position262143/context262144. Every recorded selected-row PCC passes0.995. Full attention additionally records exactly64 trailing rows for each prefill length; their minimum is0.9986376949825383. These are context checks only: this update makes no claim that the overall stage, profiling, or optimization stop gate is complete.

The former qualified v4 entry and its hashes are retained in `fused_decoder_validation_history`; earlier history is unchanged. The obsolete v5-pending entry is removed, and per-kind current fused evidence links now target the fresh results. Advertised/supported/tested capacity remains262144 with no capability reduction.

Selected source `new_recurrent_state` explicitly allocates convolution state BF16 ROW_MAJOR `[1,3,8192]`:49152 payload bytes, compared with the v4 TILE height32 payload524288 bytes, a reduction of475136 bytes. FP32 TILE recurrent state `[1,32,128,128]` remains2097152 payload bytes with the existing BF16 public consumption boundary. These payload figures exclude allocator metadata and temporary tensors. Projection-resident and grouped-down fields are retained unchanged.

CPU validation checked both source hashes; both provenance source identities; current long-context, tail-regression, decoder-factory and reference test hashes against each sidecar; all result lengths/positions/repeat counts/PCC thresholds; exact trailing-row coverage; and all current/historical linked evidence/provenance hashes. 17 linked paths were verified.

Evidence summary (hashes refer to unchanged device artifacts):

```json
{
  "linear_attention": {
    "prefill_pcc": {
      "262143": 0.9999394246973143,
      "262144": 0.9999394198457898
    },
    "decode_pcc": 0.9999835397437732,
    "provenance_timestamp_utc": "2026-11-13T06:34:27Z",
    "evidence_sha256": "ce3ea61881430f1f7cd63432bd5d9aae6f8d0479407966ae8dcb08677ef786e4",
    "provenance_sha256": "2776d3596974ba28e926e79105786ca30bafd47a98691af6df90985f8f8abc34"
  },
  "full_attention": {
    "prefill_pcc": {
      "262143": 0.9998868647370471,
      "262144": 0.9998868385799647
    },
    "decode_pcc": 0.9999496735726952,
    "provenance_timestamp_utc": "2026-11-13T06:39:09Z",
    "evidence_sha256": "5c040201500061ac85a99b8e5ef32492e32cb35c363aecba5e735a165c7a1b7f",
    "provenance_sha256": "1edfbef8aeadbb4dcfb465133aedf096fa418ccc99b15bf8a5cc3c96d6063bd9",
    "minimum_all_trailing_64_rows_pcc": 0.9986376949825383
  }
}
```

Reproduce link/hash checks without TTNN imports:

```sh
python3 - <<'PY'
import hashlib, json
from pathlib import Path
D=Path('models/autoports/qwen_qwen3_6_35b_a3b/doc')
c=json.loads((D/'context_contract.json').read_text())
def walk(x):
    if isinstance(x,dict):
        for k,v in x.items():
            if k in ('evidence','fused_evidence','provenance','decoder_source'):
                p=D/v
                assert p.is_file(),p
                if k+'_sha256' in x:
                    assert hashlib.sha256(p.read_bytes()).hexdigest()==x[k+'_sha256'],p
            if isinstance(v,(dict,list)): walk(v)
    elif isinstance(x,list):
        for v in x: walk(v)
walk(c)
assert c['supported_context']==262144 and c['capability_reduction'] is None
v=c['fused_decoder_validation']
for kind,e in v['layer_kinds'].items():
    p=json.loads((D/e['provenance']).read_text())
    assert p['source_sha256']['tt/fused_decoder.py']==v['decoder_source_sha256']
    r=json.loads((D/e['evidence']).read_text())
    assert {x['length'] for x in r['results'] if x['mode']=='prefill'}=={262143,262144}
    assert all(x['pcc']>=.995 for x in r['results'])
print('Selected v5 context links, hashes, capacity and result PCCs verified')
PY
```
