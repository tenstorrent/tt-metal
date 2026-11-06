# Client readiness sequencing correction

Contract: begin HTTP measurement only after health200. The first coordinator shell waited60seconds during B32 cache/trace startup, then incorrectly fell through to the client after the wait failed. Client reset-prefix-cache connection was refused; no measurement requests ran, all stress workers cleaned. The server continued normal warmup. Evidence: before-ready-contention.json and server/server.log.

Cause verified against shell sequencing and server timestamps. Repair: require a successful health probe before starting the client; preserve failed artifact separately. No model/harness source change or server restart. Live retry pending.
