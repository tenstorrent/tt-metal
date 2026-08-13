`tracy_ops_times.csv` is stored gzipped: the raw file was 967MB,
over GitHub's 100MB per-file limit, and a push validates every reachable blob so a later
deletion would not have helped. Content is unchanged; gunzip restores it byte for byte.
