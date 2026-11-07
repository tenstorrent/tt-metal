# Private final native inventory prepared

The helper imports the existing validated framing decoder unchanged. Run:

```sh
python3 bringup/artifacts/framed-final-inventory/inventory.py --log <completed-final-log> --output bringup/artifacts/<private-inventory>.json
```

Raw reconstructed JSON stays private; output must be under bringup/artifacts and cannot overwrite an existing file. Exit2 means coverage is compromised, even when some complete records were recovered. Native LogRecord.created remains distinct from outer first/completion/last-replay timestamps. Exact repeated frames are counted and suppressed; incomplete, malformed or conflicting records are explicitly flagged, never synthesized into completions.

Completed exact80 dry fixture42356076812 passed:60native records,32request IDs,24complete matched responses. All60envelope hashes match the existing observer's completed set; initial8request identities and full reply-body hashes match exactly.1150physicalframes consist of575unique frames plus575exact replay frames; no pending records or decoder errors. Eight unmatched requests remain listed without a failure/completion inference. Ten controlled fixtures passed for malformed/partial/unterminated frames, pending/completed conflicts, malformed reply JSON, unsupported/markerless records, duplicate replay, time separation and request matching.

This is tooling confirmation only: no scores, ongoing160 audit, client acknowledgement, freshness guarantee, dry pass or model-validity conclusion. Native request IDs do not identify task rows without separate evidence. Existing decoder bounds and conservative fragment detection remain limitations. Device cost0h; host box0.5h. No device/CI/network/dispatch, runtime change or central-ledger edit. Exact helper/decoder hashes and compact validation are in RESULT.json.

Validation interpreter: `/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/python_env/bin/python3` (Python 3.10.19). Fixture command: `python3 -m unittest discover -s bringup/artifacts/framed-final-inventory -p test_inventory.py -v` —10tests passed. Observer comparison: `python3 bringup/artifacts/framed-final-inventory/validate_completed80.py` —all assertions passed.
