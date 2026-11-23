# Fabric debug visualizer UI

## Overview

The visualizer shows what [`fabric_debug_tool`](../README.md) captured and decoded for a hung workload. You launch it  
with `serve.py`:

```bash
python3 tt_metal/fabric/debug/visualizer/ui/serve.py --files-path <logs folder>
```

`serve.py` finds the captures under the folder and serves the page. The page loads the decoded files. It reads the
`.bin` files only on demand, a byte range at a time.

What each part of the page shows is under [Using the page](#using-the-page), after the server's endpoints.

## Local and remote mode

The server is the same in both modes. It always listens on `127.0.0.1` only. The modes differ in what `serve.py` does
once it is listening.

**Local** is the default. Use it when the captures are on the machine you are sitting at. `serve.py` prints the URL and
opens your browser:

```
$ python3 tt_metal/fabric/debug/visualizer/ui/serve.py --files-path /path/to/logs
fabric debug visualizer: 2 captures under /path/to/logs
  20261113T230100Z  1 run, 2 ranks  /path/to/logs/torus2rank/generated/fabric_debug/runs
  20261113T220213Z  1 run, 1 rank  /path/to/logs/ringbubble/generated/fabric_debug/runs

Open http://localhost:8765/
Ctrl-C stops the server.
```

**Remote** is for captures on a host you reached over ssh, where a browser on the host isn't useful. `--remote`
takes no value. Instead of opening a browser, `serve.py` prints an `ssh` command to run on your own machine, and the
URL to open once it is running:

```
$ python3 tt_metal/fabric/debug/visualizer/ui/serve.py --files-path /path/to/logs --remote
fabric debug visualizer: 2 captures under /path/to/logs
  20261113T230100Z  1 run, 2 ranks  /path/to/logs/torus2rank/generated/fabric_debug/runs
  20261113T220213Z  1 run, 1 rank  /path/to/logs/ringbubble/generated/fabric_debug/runs

On your own machine, forward the port:

    ssh -N -L 8765:localhost:8765 me@lab-host

  through a jump host, add -J <jump host>, for example: ssh -N -J exabox -L ...
  if 8765 is taken on your machine, change the first 8765 and the URL's port (in the above ssh   command) to match

then open http://localhost:8765/
Ctrl-C stops the server.
```

Run the printed `ssh` command in a second terminal on your own machine and leave it running, then open the URL in your
browser.

### The tunnel

The server listens only on the lab host's `localhost`, so nothing else on the network can reach it. The `ssh -L`
command makes it reachable from your machine: your machine's `localhost:8765` is carried over the SSH connection to the host's `localhost:8765`. `-N` means forward only, with no shell on the other end.

- **Host name:** the user and host in the command are the account you are logged in as and the machine's own hostname.
That may not be the name you use to ssh to it, such as an alias in `~/.ssh/config`. If so, replace the host.
- **Jump host:** a host you reach through another one needs `-J <jump host>` in the command, for example
`ssh -N -J kix1 -L 8765:localhost:8765 me@node-name` for a KIX1 node.
- **Port in use on your machine:** change the first port number in `-L` and the port in the URL to a free one. The
second port number stays as `serve.py` printed it.

## Arguments

| Argument | Default | What it does |
| --- | --- | --- |
| `--files-path <folder>` | `TT_METAL_LOGS_PATH` | The folder to search for captures. See the next section. `serve.py` exits with an error if this isn't given and `TT_METAL_LOGS_PATH` isn't set, or if the folder doesn't exist |
| `--port <port>` | 8765 | The port to listen on. If it is taken, `serve.py` tries the next 9 ports and prints the one it used. It exits if all 10 are taken |
| `--remote` | Off | Prints the `ssh` command that forwards the port, instead of opening a browser. See the previous section |
| `--no-browser` | Off | Doesn't open a browser. Local mode only: `--remote` never opens one |

## Where it looks for captures

`--files-path` can be:

- **A logs folder**, such as `TT_METAL_LOGS_PATH`. `serve.py` searches `generated/fabric_debug/runs/` in it.
- **A folder** `fabric_debug_tool_copy.py` **made**, `<dest>/<capture_id>/`, which holds a folder for each run.
- **A folder that holds several logs folders**, such as one folder per machine or per run of a test.
- **A single run folder or capture folder.**

A capture is found by its decoded files, `captures/<capture_id>/fabric_decoded_rank_<r>_of_<n>.json`. The rank's
`.bin`, `fabric_capture_rank_<r>_of_<n>.bin`, is the file beside it, when there is one. The search skips folders and
files whose names start with `.`, goes at most 10 folders below `--files-path`, and stops after 20000 folders. When it
stops, `serve.py` says so and suggests a `--files-path` closer to the captures. For the layout of a logs folder, see
`fabric_debug_tool`'s [README](../README.md).

The list is searched again each time the page asks for it, so a capture made while the server is running shows up when
you reload the page. The list printed at startup is the one from when the server started.

The captures are listed newest first, by capture id, which is a UTC time.

## Endpoints

| Request | Returns |
| --- | --- |
| `GET /` | The UI's files, `index.html` by default |
| `GET /api/captures` | Every capture found, as JSON: each capture's id and folder, its runs, and each run's ranks with the name, URL and size of the rank's decoded file and `.bin` |
| `GET /files/<path>` | A decoded file or `.bin` that `/api/captures` listed, by the path it gave |

`/files/` serves only the files `/api/captures` listed, looked up by exact path, so no other path on the host can be
read through it. It handles `Range` requests, which is how the page reads part of a `.bin`. It sends decoded files
gzip-compressed when the browser accepts it.

The server logs errors only, to stderr.

## Using the page

### Choosing a capture

The page lists the captures the server found, newest first, with each one's time, run, ranks and file sizes. When
there is only one, it opens straight away. The page can also open a folder from your own machine, picked or dropped on
it: a logs folder, a run folder or a capture folder. The browser reads that folder itself, without the server.

### Loading and the summary

Opening a capture reads every rank's decoded file and checks that they are decoded files this page can read, from the
same capture. A missing rank or `.bin` doesn't stop it: the chips of a missing rank show without routers, and a rank
without its `.bin` can't show raw bytes.

A summary of the capture then shows above the panels: when and where it was captured, the build, the fabric config,
the hardware, how much was read, and decode's router verdicts and findings. Notes below it list any warnings, such as
a missing rank or a core UMD's discovery skipped.

### Hardware view

The hardware view draws every mesh of the capture in one view, chips placed by their mesh coordinates.

- **Chips:** each chip takes the colour of the worst verdict among its routers: stuck, then not looping, unknown,
  looping and running. A badge counts the chip's findings. A dashed outline means not every router on the chip was
  read, and a dotted grey chip belongs to a rank that isn't loaded.
- **Links:** a line joins each pair of linked chips, whatever the number of routing planes between them. Wrap links run
  around the outside of their mesh, and links between chips of different MPI ranks are drawn in their own colour.
- **Routing plane:** pick a plane to see only its routers and links.
- **Zoom and pan:** scroll or pinch to zoom, drag to pan, and Fit to see everything again.
- **Details:** hover a chip for its details, and click it to open it in the chip view below.

### Chip view

The chip view draws one chip's die with its routers on the side each one faces, one per routing plane. Pick the chip
from the list, or click a chip in the hardware view.

- **Routers:** each is labelled with its name and Ethernet channel and outlined by its verdict. An orange dot marks a
  router on a core UMD's discovery skipped.
- **Inside the die:** for the selected plane, an arrow from each router's receiver to each router it can forward into
  (`intra_chip_downstream_edges`). An arrow is solid when the router forwards there (`live_downstreams`), and dashed
  when it's only connected. Hover an arrow for the sender channel it feeds and that channel's free slots.
- **Around the die:** each side's label names the chip and routers it links to. Click it to open that chip.
- **Table:** every router's direction, plane, channel, peer, link, verdict, capture status and findings. Click a row or
  a router to open it in the router view below.

### Router view

The router view shows one router: its facts, its activity, and its channels with what the capture read about them.
Every count is shown as a count. The capture doesn't say which slots hold a packet, so every slot is drawn the same,
and a bar only shows how many slots are in use.

- **Facts and activity:** direction, plane, Ethernet channel, cores, peer and link, then the verdict and every heartbeat
  read behind it, round by round. A heartbeat that didn't move is shown in red.
- **ERISCs:** one subsection per ERISC the router's kernel runs on, with what the decoded file records for it under its
  own key names: `processor`, `physical_erisc` and `opt_level` (the level the kernel was compiled at).
- **Packet slot:** the header, the largest payload and the slot size, drawn to scale.
- **Sender cards:** who feeds the channel (a worker, or a sibling router), the packets not sent yet (`free_slots`),
  the packets sent but not credited back to the producer (`uncredited_at_peer`, and how decode worked it out), and the
  completions from the peer: the running total of the channel's packets the peer's receiver has finished with, as
  this router's `to_sender_completion` beside the peer's `receiver_completion`. Then the producer's connection
  (`connection`, and `conn_info`'s `worker_xy` and `edm_read_counter`) and, on the worker channel, the producer's write
  cursor (`buffer_index_sem`).
- **Receiver cards:** the packets held (`pkts_sent`) and when the receiver takes them, the completion group, the
  credits sent back for each of the peer's sender channels beside what the peer received, and each sibling channel it
  forwards into, with that channel's free slots.
- **Slots:** one row of boxes across each card, below its facts and above its fields. A hollow slot was never written:
  its header is all zeros. Hover one for its address and the header last written to it, which may be from a packet
  that's gone, and click it to open that header (see Packet headers and paths below).
- **Slot states:** on a router run with `TT_METAL_FABRIC_ENABLE_CHANNEL_PROGRESS_LOG=1`, decode says which slots hold
  a packet (each channel's `analysis.slots`). Each slot is then shaded:
  - teal: pending, written but not sent (sender) or arrived but not sent on (receiver);
  - blue: held, sent and awaiting credit;
  - grey: free.

  The next slot is underlined. A striped slot with an amber border is counted as both pending and held: the log and
  the count register disagree. On a running router they were read apart; on a stopped one, the router's own counts
  disagree. A line under the slots counts each state, or says why decode couldn't tell. The card also lists the
  logged progress: `sent`, `completed` and `first_level_acked` for a sender, and `next_index`, `uncredited` and
  `packets_ready` for a receiver. The router's facts say whether the log was on. Under the slots, a table counts
  where the channel's headers go: the router that reads each packet next, what it does there, and where it goes after.
- **Credit counters:** the router's four credit counter arrays. The columns a channel uses are bright.
- **Fields:** every field, per ERISC, grouped by category, in each card and for the router. Fields the builder didn't
  allocate are listed under Not allocated.
- **Routing table:** at the bottom, `routing_l1_info_t` as decode reads it, as text. The router's own
  `state_manager` and which copy of the chip's table it holds, then the chip's table: its identity beside the chip's,
  `intra_mesh_directions` as one row per destination chip, the directions and exit chips
  toward other meshes, and in 1D `path_table_1d` (each entry's index, hop count and actions), in 2D `action_vectors` as two
  matrices with this chip's row and column highlighted and `multicast_trees` as parent, direction and child. Only the
  parts the fabric config fills are shown.

Click a peer, a sibling or a downstream router to open it.

### Packet headers and paths

Clicking a slot opens its header in the card, under the decoded file's key names:

- **Header:** the send type, payload size and source channel.
- **Command:** the NoC command, with each NoC address as (x, y) and an address.
- **Route:** in 1D, one action per hop. In 2D, the destination, the multicast hop counts and the route's `y` (by row)
  and `x` (by column) entries.
- **Path:** each node of the route, from the receiver that reads the packet next: on a sender's slot that's the peer
  across the link, on a receiver's it's this router. Each node has its router, VC, the node it came from and the
  direction, which map it reads in 2D, its action, and whether it delivers or where the branch ends. Below each node,
  that router's verdict, and a warning when a direction the action forwards in isn't live there or the receiver has
  forwarding disabled. Decode names routers on its own rank only, so a node on another rank's chip shows the chip.

**Show on hardware** draws the path on the hardware view: an arrow per hop (round the wrap when the hop takes one), a
dashed arrow across the link for a packet a sender hasn't sent yet, a dot on each chip the packet is delivered to, and
a red mark where a branch ends without delivering. A multicast draws as a tree. The banner above the view names the
slot, scrolls back to it, or clears the path.
