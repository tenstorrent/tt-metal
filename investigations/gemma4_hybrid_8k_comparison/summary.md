Legacy: one 8192-token request. Hybrid: 512 + 512 + 1024 + 1536 + 2048 + 2560 = 8192 tokens.

Median device time over 5 warmed replays, including gaps between programs:

|   Prefix per request |   Legacy ms |   Hybrid ms |
|----------------------|-------------|-------------|
|                    0 |      111.13 |      160.11 |
|              253,952 |      268.55 |      564.69 |

Each request uses the indicated prefix. Loading, compilation, capture, input/metadata staging, and warmup replays are excluded. KV history is synthetic.
