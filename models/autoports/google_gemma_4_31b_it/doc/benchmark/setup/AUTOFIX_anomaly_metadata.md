# AutoFix: tokenizer prompt metadata serialization

Observed execution failure: setup/metadata-failed-anomaly_run.log records a BatchEncoding JSON serialization TypeError after anomaly_replies/0-response.json was safely saved. The installed tokenizer returns a mapping from apply_chat_template(tokenize=True). Diagnosis is directly established by the exception and API result; the serving request succeeded.

Repair: extract input_ids from mappings before JSON serialization. Resume validates and reuses each existing request/response pair, rather than rerunning it. Future reply wall timing is saved before downstream metadata construction. The first reply elapsed time was not durably saved and stays null with an explicit explanation; no timing is reconstructed from filesystem timestamps.

Validation: a host-only tokenizer probe verifies plain input_ids and JSON serialization. The resumed anomaly run and final prompt-token-count comparisons supply integration evidence. No model, precision, sampler, plugin or serving runtime change.

Observed host probe: encoding type BatchEncoding; extracted reference55tokens equals served55tokens for saved reply0, which contains1413output tokens. Resumed replies0–2 all have matching reference/served prompt counts and no detector flag. Original response0 was reused, not regenerated.
