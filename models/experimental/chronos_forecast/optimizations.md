## model optimzations
- group attention: 965 ms original impl
- Series sort by group, pack into tile aligned blocks and run group attention one small per block instead of one large 1024x1024(155 -> 45ms) on forward
- DRAM swap batch and time axses through row major (2x) over tile permute

- l1 chunks interleaved, not block shareded
- group blocks 32 to 128(not one per group)


## sweeps
sweep_matul.py
sweep_sdpa.py
sweep_l1_chunk.py
