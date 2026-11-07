# Static diagnosis: effort ledger exceeds file-size hook

Coordinator reported a commit failure. Independently reproduced with python3 scripts/check_file_size.py on the current ledger before editing. The configured local check-large-files hook invokes this script with its default500KB threshold. The script calculates ceil(bytes/1024) and rejects values above500; the exact permitted maximum is512000bytes. Current measured byte count and failing output are in private check.json.

This is serialized-document size, not a field validity or runtime failure. JSON whitespace outside quoted strings consumes removable space. A lexical compaction preserving strings, numbers, key order and all data is sufficient; no archive, history deletion, threshold override or source change is needed. This report is direct bounded source diagnosis and reproduction, not a claim of a separately launched investigator.
