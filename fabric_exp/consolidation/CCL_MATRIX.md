# DeepSeek CCL trace perf matrix: main's router vs v4 (2026-11-02)

Script: models/demos/deepseek_v3_b1/scripts/ccl_trace_perf_matrix.py, run per router by ccl_matrix_runs.sh (P150x8,
build_ttnn, TT_FABRIC_NO_BW_TELEMETRY=1). Routers: main_fw (kernel v1 as shipped), v4rxo0b (the lead + inline-write
fix), v4rxf4 (receiver lead candidate). Payloads 2048/4096/4352/8192/15232; SDPA at its tuned (num_l_chunks,
compute_block_size) only. Passes a (main_fw first) and b (reversed); SDPA 2048 B also c/d/e (rotated order).
Op time in us, mean over passes, each pass in brackets; ratio v4/main per v4 router. Table: `ccl_matrix_report.py a b`
(and `a b c d e` for the SDPA 2048 B row). Jobs 430, 431, 435, 440-444.

```
ccl                  config payload                main_fw                 v4rxf4                v4rxo0b   v4/main
all_gather                1    2048    2.992 (2.995/2.988)    2.862 (2.861/2.863)    2.859 (2.864/2.854)   0.957 0.956
all_gather                1    4096    2.640 (2.654/2.627)    2.533 (2.536/2.530)    2.534 (2.540/2.529)   0.959 0.960
all_gather                1    4352    2.628 (2.637/2.620)    2.521 (2.525/2.517)    2.534 (2.523/2.544)   0.959 0.964
all_gather                1    8192    2.633 (2.623/2.642)    2.534 (2.531/2.537)    2.540 (2.535/2.546)   0.963 0.965
all_gather                1   15232    2.642 (2.633/2.651)    2.522 (2.520/2.523)    2.526 (2.525/2.528)   0.954 0.956
all_reduce                1    2048    1.438 (1.438/1.437)    1.429 (1.429/1.429)    1.427 (1.429/1.426)   0.994 0.993
all_reduce                1    4096    1.251 (1.247/1.255)    1.259 (1.255/1.263)    1.252 (1.252/1.252)   1.006 1.001
all_reduce                1    4352    1.288 (1.274/1.303)    1.254 (1.257/1.251)    1.257 (1.254/1.261)   0.973 0.976
all_reduce                1    8192    1.280 (1.279/1.282)    1.253 (1.255/1.252)    1.254 (1.254/1.255)   0.979 0.980
all_reduce                1   15232    1.281 (1.279/1.284)    1.261 (1.259/1.264)    1.260 (1.257/1.262)   0.984 0.983
all_reduce                2    2048    1.373 (1.370/1.376)    1.339 (1.341/1.336)    1.339 (1.337/1.341)   0.975 0.975
all_reduce                2    4096    1.247 (1.245/1.249)    1.250 (1.248/1.252)    1.244 (1.245/1.242)   1.002 0.997
all_reduce                2    4352    1.270 (1.266/1.274)    1.253 (1.254/1.253)    1.248 (1.248/1.247)   0.987 0.982
all_reduce                2    8192    1.271 (1.267/1.275)    1.252 (1.251/1.254)    1.252 (1.255/1.249)   0.985 0.985
all_reduce                2   15232    1.282 (1.281/1.283)    1.250 (1.245/1.255)    1.250 (1.249/1.251)   0.975 0.975
broadcast                 1    2048    6.349 (6.348/6.351)    6.300 (6.296/6.303)    6.300 (6.290/6.309)   0.992 0.992
broadcast                 1    4096    5.133 (5.153/5.114)    5.079 (5.084/5.073)    5.072 (5.078/5.067)   0.989 0.988
broadcast                 1    4352    5.116 (5.109/5.123)    5.067 (5.071/5.063)    5.080 (5.080/5.080)   0.990 0.993
broadcast                 1    8192    4.226 (4.227/4.224)    4.196 (4.200/4.191)    4.186 (4.188/4.185)   0.993 0.991
broadcast                 1   15232    3.893 (3.889/3.898)    3.869 (3.869/3.869)    3.869 (3.867/3.872)   0.994 0.994
broadcast                 2    2048    9.219 (9.223/9.214)    9.139 (9.167/9.110)    9.107 (9.112/9.102)   0.991 0.988
broadcast                 2    4096    7.950 (7.981/7.919)    7.867 (7.863/7.870)    7.859 (7.868/7.849)   0.989 0.988
broadcast                 2    4352    8.005 (7.997/8.012)    7.885 (7.888/7.881)    7.885 (7.891/7.878)   0.985 0.985
broadcast                 2    8192    7.049 (7.039/7.059)    6.968 (6.961/6.974)    6.960 (6.954/6.966)   0.988 0.987
broadcast                 2   15232    4.155 (4.158/4.152)    4.119 (4.115/4.123)    4.121 (4.116/4.127)   0.991 0.992
sdpa_reduce_to_all    l2,b4    4096    5.614 (5.615/5.612)    5.415 (5.471/5.360)    5.447 (5.452/5.442)   0.965 0.970
sdpa_reduce_to_all    l2,b4    4352    5.596 (5.570/5.621)    5.469 (5.479/5.458)    5.441 (5.403/5.479)   0.977 0.972
sdpa_reduce_to_all    l2,b8    8192    5.644 (5.665/5.623)    5.428 (5.446/5.409)    5.482 (5.466/5.497)   0.962 0.971
sdpa_reduce_to_all    l2,b8   15232    6.318 (6.307/6.329)    5.478 (5.469/5.486)    5.461 (5.456/5.466)   0.867 0.864
sdpa_reduce_to_all    l4,b8    2048    5.880 (5.687/5.695/5.685/6.285/6.047)    5.489 (5.687/5.687/5.397/5.402/5.271)    5.341 (5.258/5.411/5.379/5.395/5.263)   0.934 0.908
```

## Reading

- v4 is faster than main on every CCL: all_gather -3.6..-4.6%, all_reduce -0.6..-2.7% (flat at 4096 B),
  broadcast -0.6..-1.5%, SDPA -2.3..-3.8% at 4096-8192 B and -13.5% at 15232 B.
- SDPA's op time is bound by the compute core (TRISC_2) on every router, so the router shows through data arrival.
- SDPA 15232 B: main 6.31/6.33 against its own 5.64 at 8192 B; v4 flat (~5.47). Cause not examined.
- SDPA 2048 B is noisy on every router (main 5.69-6.29, v4rxf4 5.27-5.69, v4rxo0b 5.26-5.41 over five passes; the
  values cluster, e.g. 5.687 appears on main and v4rxf4). Medians: main 5.69, v4rxf4 5.40, v4rxo0b 5.38.
- v4rxf4 = v4rxo0b within noise on every cell: the receiver chain (incl. the codec stage) does not show in CCL op times.

## Not covered

- reduce_to_one: its trace benchmark captures 180 traces of 8 KiB into a 1 MiB trace region (128 fit); it fails on
  every router before timing. Pending the user's choice (2 MiB region by a local test edit, or fewer samples).
- SDPA needed a local test edit (test_sdpa_reduce_to_all.py: output_l_mesh tile=None; the row-major page config
  rejects a custom tile). Revert after the matrix.
