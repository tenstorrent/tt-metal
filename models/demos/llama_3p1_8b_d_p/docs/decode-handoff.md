# Prefill/decode cache contract

See the [KV-chunk table contract](runner-integration.md#kv-chunk-table-contract)
for config order, page size, owners, rotary coordinates and slot semantics.

The shared producer verifies source K/V through the exported table and device map.
The receiver must additionally agree on the model, checkpoint, value format, valid
length, request ownership, completion signals and allocation lifetime. Cross-host
transfer and decode integration require separate end-to-end validation.
