# Held C4/80 fast-forward fallback

Local candidate `9a874b6574c157e77e5f4445e3134885589af96d` directly descends published `7ad6cf50d160a2e9cbe007fc4475af3c8f8bf365`. Production eval configuration and model YAML exactly match old C4/80 `1eca89c0dcf65550a79196aecca13ef687c9cf06`: frozen80, C4/B8, no expandedpool environment, accepted response/request logging, unchanged limits/sampling/seed/runtime. The only difference from the entire old C4/80 tree is the retained/adapted7-line test confirming default pool. All unrelated7ad source preserved.

36 focused host tests passed in3.89s; full hooks, exact AST/tree comparison and whitespace check passed; worktree clean. Device0, hostbox0.3h. Reuses accepted ab273/0696f29a image under existing proof; no new image/runtime changes or measurement.

Held until valid40 and measured projection; no dependency on finishing the C8 comparison. Coordinator chooses current best at the boundary. No push/dispatch/central edits; prior candidates retained. No score or projection claimed.
