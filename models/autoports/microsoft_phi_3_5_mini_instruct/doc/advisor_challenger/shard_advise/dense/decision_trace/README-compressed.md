# decode_decision_trace.json is stored gzipped

The advisor's decision trace for this capture was **112.56 MB**, over GitHub's hard 100 MB per-file
limit, and a push validates every reachable blob — so the cell could not be published at all with the
raw file in its history. It is stored here as `decode_decision_trace.json.gz` instead; the content is
unchanged, `gunzip` restores it byte for byte.

The oversized blob was removed from this branch's history (the driver's commit was rewritten), not
merely deleted in a follow-up commit, because a later deletion does not stop the push from validating
the blob that is still reachable.
