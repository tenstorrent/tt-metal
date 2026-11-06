# Offline analyzer correction

Removed intermediate pending_incomplete from historical loss aggregation; permanent loss/disable/export-failure/file-exhaustion/truncation remain checked across snapshots. Final pending_steps and final token_retention.pending_incomplete remain explicit closure fields. Preserved original analysis.json, wrote analysis-final.json; finalized RESULT was not rewritten.

Focused existing-evidence checks passed:592 intermediate pending snapshots; zero permanent-loss flags; final pending0/pending_incompletefalse; request chains and repetition metrics byte-equivalent between analyses; corrected historical lossfalse versus originaltrue. Host-only correction,0devicehours, no replay/re-dispatch.
