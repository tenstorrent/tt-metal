# DeepSeek-V4.1-Flash, disaggregated: set up, build, run

Prefill and decode run on two Blackhole Galaxies (8x4, 32 chips each), one host each. The prefill server
(`demo/prefill_server.py`) runs on one. The decode server (`demo/server.py`) runs on the other, sends every
prompt to the prefill server, and loads the KV cache that comes back. Two ways to move that cache:

| Mode | How the KV cache moves | How you launch | What you need |
| --- | --- | --- | --- |
| **fabric push** | prefill writes it into decode's DRAM over TT-Fabric; TCP carries only control messages | both servers as the two ranks of one `tt-run` job (`demo/run_d_p_servers.sh`) | the Galaxies cabled to each other, and section 4 passing |
| **TCP** | prefill reads it back to its host, sends it over TCP, decode writes it onto the device | two processes, one per host | an IP route between the hosts |

At 1M tokens the push finishes in ~1.4 s after the prefill. TCP takes ~14 s. Use TCP when section 4 fails, when
the Galaxies are not cabled to each other, or when you need to start and stop the two servers separately.

Details: `PREFILL_D_P_FABRIC_PUSH.md` (push), `PREFILL_D_P_INTEGRATION.md` (TCP),
`PREFILL_D_P_PREFIX_CACHE.md` (prefix cache). One host and one Galaxy: `QUICKSTART.md`.

## 1. Name your hosts

Set these in every shell you run the commands from:

```bash
export PREFILL_HOST=<prefill host, as ssh and MPI reach it>   # rank 0
export DECODE_HOST=<decode host, as ssh and MPI reach it>     # rank 1
export PREFILL_IP=<prefill host's IP>                         # decode connects here
export DECODE_IP=<decode host's IP>                           # clients connect here
export MPI_IFACE=<network interface MPI uses, on both hosts>
export REPO=<checkout path; the same path on both hosts>
export PREFILL_PORT=9041 DECODE_PORT=8000
```

Check:

* `$MPI_IFACE` is up on both hosts (`ip -br addr show $MPI_IFACE`) and each host can reach the other on it.
* From the prefill host, `ssh -o BatchMode=yes $DECODE_HOST hostname` prints the decode hostname without a
  password prompt. `tt-run` connects to the decode host over ssh from the prefill host.
* On the decode host, `getent hosts $PREFILL_HOST`. If the decode host cannot reach that address, use
  `$PREFILL_IP` in the prefill URLs below.
* Someone else may be using the cards. Before a command that opens a device:
  `ps -eo user,pid,args | grep -E "tt-run|server.py|pytest"`.

## 2. Set up and build, on both hosts

`tt-run` runs the launch host's paths on every rank. Unless `$REPO` is on storage both hosts mount, each host
needs its own checkout of the same commit, at the same path, and each checkout has to be built.

```bash
cd $REPO
git checkout <the branch both hosts will run>
sudo ./install_dependencies.sh            # first time on this host
./build_metal.sh                          # again after C++ changes; files under kernels/ compile at runtime
./create_venv.sh                          # first time on this host
source python_env/bin/activate
```

On the decode host, build the Rust host runtime. The decode server runs its per-step host work there. Without
the build it uses slower Python threads and starts no web monitor. If `cargo` is missing:
`curl https://sh.rustup.rs -sSf | sh -s -- -y && source ~/.cargo/env`.

```bash
models/experimental/deepseek_v41_flash/host_rs/build.sh
```

Download the checkpoint on both hosts. It is ~480 GB. Both servers read the default Hugging Face cache,
`~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017`.

```bash
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017
```

On the decode host, fill the expert weight cache before the first server start. Otherwise that start converts
the experts one at a time. Host only, no device. The cache is ~700 GB.

```bash
python models/experimental/deepseek_v41_flash/tools/fill_expert_cache.py --all-layers
python models/experimental/deepseek_v41_flash/tools/fill_expert_cache.py --dspark --dtype bfloat8_b
```

The prefill host needs ~210 GB of free RAM for the Engram lookup tables. The first start builds them into
`/dev/shm/deepseek_v41_d_p_engram` (`ENGRAM_SHM_DIR`). Later starts map that directory.

### Keep the two checkouts on the same commit

Commit on one host, then fast-forward the other over ssh. This example runs on the prefill host and pulls the
decode host's commits. The prefill working tree has to be clean, and `$REPO` has to be an absolute path:

```bash
git fetch ssh://$DECODE_HOST$REPO <branch>:refs/remotes/decode/<branch>
git merge --ff-only decode/<branch>
```

If both hosts have commits the other does not, merge. Then `./build_metal.sh` on any host whose C++ changed.
Before a launch:

```bash
git rev-parse HEAD; ssh $DECODE_HOST "git -C $REPO rev-parse HEAD"
```

## 3. Shell variables

On the prefill host, from `$REPO`, with the venv active. Section 4 uses `$CFG` and `$MPI`.

```bash
CFG=tests/tt_metal/distributed/config
printf 'rank 0=%s slot=0\nrank 1=%s slot=0\n' $PREFILL_HOST $DECODE_HOST > /tmp/d_p.rankfile
MPI="--host $PREFILL_HOST,$DECODE_HOST --map-by rankfile:file=/tmp/d_p.rankfile \
  --mca btl self,tcp --mca btl_tcp_if_include $MPI_IFACE --bind-to none --tag-output"
```

The rank-binding files in section 4 have `40_41` in the name. That is the pair of machines they were first
written for. The files do not name your hosts. The rankfile above does: rank 0 on `$PREFILL_HOST`, rank 1 on
`$DECODE_HOST`.

## 4. Test the link between the Galaxies

Run level 1 and level 2 before the first fabric-push launch, and again after you re-cable, run
`tt-smi -glx_reset`, or a push fails. Each command exits 0 when it passes. `tt-run` prefixes every line with
`[job,rank]`. Rank 0 is the prefill host, rank 1 the decode host. With a single job those prefixes are `[1,0]`
and `[1,1]`.

If level 1 or level 2 fails, serve with section 6. Level 3 checks the TCP path that section 6 uses.

### Level 1: traffic across the cables

Both Galaxies as one 16x4 mesh (8 rows on each host), and an `all_reduce` across the host boundary.

```bash
tt-run --rank-binding $CFG/dual_8x4_hosts_40_41_rank_bindings.yaml --mpi-args "$MPI" \
  bash -c 'cd $TT_METAL_HOME && exec python tests/tt_metal/multihost/test_fabric_cross_host.py'
```

Pass: both ranks print `ALL CHECKS PASSED`.

### Level 2: the write the servers do

Two meshes, one Galaxy each, with the servers' 6208-byte packet size. Rank 1 allocates decode-shaped KV pools
and sends their addresses to rank 0. Rank 0 writes them over the fabric. Rank 1 checks the bytes. Keep
`--payload 6208`.

```bash
tt-run --rank-binding $CFG/dual_bh_galaxy_8x4_hosts_40_41_two_mesh_rank_bindings.yaml --mpi-args "$MPI" \
  bash -c "cd \$TT_METAL_HOME && D=models/experimental/deepseek_v41_flash/tests/xhost_kv
    if [ \"\$OMPI_COMM_WORLD_RANK\" = 0 ]; then exec python \$D/sender_write.py --payload 6208; fi
    exec python \$D/receiver_init.py --sender $PREFILL_IP --payload 6208"
```

Pass: rank 1 prints `PASS`. Then, on the decode host, after that job has exited, check the bytes are still in
DRAM. This command does not open the device:

```bash
python models/experimental/deepseek_v41_flash/tests/xhost_kv/receiver_verify.py   # prints PASS
```

A timeout while the fabric is coming up, a rank reporting links down, a hang, or `FAIL`: use section 6. If a
hang drops a chip off PCIe, `tt-smi` on that host shows fewer than 32 chips. `tt-smi -glx_reset` there brings
them back, then rerun this section.

`tests/xhost_kv/d_p_push.py` loads both models and compares a real push with the TCP path. Sections 5 and 6 run
without it. The launch command in its docstring names the original 40/41 hosts, and the run takes several
minutes. Rank 1 prints `PASS: byte-exact`.

### Level 3: TCP

Start a prefill server with section 6. On the decode host, venv active (torch only, no device):

```bash
python models/experimental/deepseek_v41_flash/demo/prefill_client.py --server $PREFILL_IP:$PREFILL_PORT --random 16384 --repeat 3
```

Pass: it prints `PASS`. With no server up, `nc -zv $PREFILL_IP $PREFILL_PORT` only tells you the port is open.

## 5. Run fabric push

After section 4 passes. On the prefill host, from `$REPO`, venv active:

```bash
models/experimental/deepseek_v41_flash/demo/run_d_p_servers.sh --prefill-port $PREFILL_PORT -- \
  --host 0.0.0.0 --port $DECODE_PORT --prefill-url $PREFILL_IP:$PREFILL_PORT --kv-transfer-mode fabric
```

The script reads `PREFILL_HOST`, `DECODE_HOST`, and `MPI_IFACE`. Rank 0 is the prefill server, rank 1 the
decode server, and they are one job: Ctrl-C on the prefill host stops both. Decode retries the prefill server
for up to two hours (`DEEPSEEK_V41_PREFILL_WAIT`) while prefill loads weights and warms up.

At the default 1M-token context, a cold start is about 200 s of weight load, 160 s of first-run compile, and
22 s of warmup. The prefill server is listening when it logs `prefill server on ...`. Decode then logs
`disaggregated prefill connected to ... KV pushed over TT-Fabric`. It is ready for requests when it logs
`serving 'deepseek-v4.1-flash' on http://...`.

## 6. Run TCP

Two processes. No `tt-run`. Venv active in both shells.

Prefill host:

```bash
python models/experimental/deepseek_v41_flash/demo/prefill_server.py --port $PREFILL_PORT
```

Decode host. `--prefill-wait 7200` is the same two-hour retry as section 5:

```bash
python models/experimental/deepseek_v41_flash/demo/server.py --host 0.0.0.0 --port $DECODE_PORT \
  --prefill-url $PREFILL_IP:$PREFILL_PORT --kv-transfer-mode tcp --prefill-wait 7200
```

You can restart either process on its own. Decode connects again on the next prompt.

## 7. Send a request

From a machine that can reach the decode host (`$DECODE_IP` and `$DECODE_PORT` set there too):

```bash
curl -s --no-buffer -X POST http://$DECODE_IP:$DECODE_PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"messages": [{"role": "user", "content": "Hello"}], "stream": true}'
```

The web monitor is `http://$DECODE_IP:8090` (prefill progress, decode tok/s, KV pools) when the Rust runtime
from section 2 is built. The page stays up after the decode server exits. Use the curl above to see if the
server is still up.

Prompt-length sweep:

```bash
python models/experimental/deepseek_v41_flash/demo/perf_client.py --url http://$DECODE_IP:$DECODE_PORT --min-len 32k --decode-tokens 3072
```

One measured fabric-push run, one user, DSpark on (the decode server's default; `--no-dspark` turns it off):

| length | TTFT (s) | prefill tok/s | decode tok/s |
| --- | --- | --- | --- |
| 32K | 1.40 | 23,470 | 189.5 |
| 128K | 3.48 | 37,668 | 183.7 |
| 512K | 12.09 | 43,365 | 165.0 |
| 1M | 26.44 | 39,542 | 153.0 |

## 8. Troubleshooting

| What you see | What to do |
| --- | --- |
| `tt-run` cannot reach the decode host | Passwordless ssh from the prefill host to `$DECODE_HOST` (section 1). |
| Fabric init times out, or a rank reports links down | Rerun section 4, level 2. If it still fails, use section 6. |
| Import error, or an op one host has and the other does not | The checkouts differ, or one host was not rebuilt (section 2). |
| `prefill server ... not listening yet; retrying` | Prefill is still loading or warming up. Decode gives up after `--prefill-wait` seconds. |
| Decode cannot connect to the prefill server | The hostname in the URL resolves to an address decode cannot reach. Use `$PREFILL_IP`. |
| `tt-smi` shows fewer than 32 chips after a hang | `tt-smi -glx_reset` on that host, then section 4. |
| The first prefill start spends many minutes building Engram tables | Expected. They land in `/dev/shm/deepseek_v41_d_p_engram`. The next start maps them. |
