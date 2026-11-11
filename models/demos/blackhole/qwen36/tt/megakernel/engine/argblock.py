"""Per-core arg blocks for the megakernel (spec v1.25; device side: attn/kernels/mk_argblock.hpp).

Why: dispatch sizes the runtime-arg region and the CB-config block program-wide (largest core, max CB index), and STREAM
is ~40 B from the 70656 B kernel-config limit. So per-core tables travel as data:

  * one HEIGHT_SHARDED L1 uint32 tensor per KIND over the cores that read it, shard = one core's block;
  * the shard address is identical on every core (and every device), so it goes in ONE common runtime arg, index = kind id;
  * every kernel of the engine program carries the same common args ``common_args(blocks)`` ([addr of kind 0, 1, ...]);
  * a kernel reads its block once at init: ``mk::argblock(mk::AB_AR)[i]``.

Block = [magic "MKAB", kind id, n payload words, version, payload..., zero pad to a multiple of 4 words (16 B)]. All cores
of one kind get the same block width W (the max payload over cores, padded). Per-device tables (e.g. AR roles differ by
ring position) are built per device and assembled with from_host_shards: same shape, same address, different bytes.

Also here: ``check_cb_ids`` (CB ids <= MAX_CB_ID keep the program-wide CB-config block small; 16 B per index on every core).
"""

MAGIC = 0x4D4B4142
HEADER = 4
KINDS = {"phase": 0, "ar": 1, "v12": 2, "gdn": 3, "attn": 4}  # = common runtime arg slot; mk_argblock.hpp AB_*
MAX_CB_ID = 48  # spec v1.166: 44 -> 48 for G-b's E1_GB prefix CBs 45-48 (+64 B config on every core; fit proven on the mesh)


def _cores(crs):
    import ttnn

    return list(ttnn.corerange_to_cores(crs, row_wise=True))


def block_words(kind, payload, width, version=1):
    """One core's block: header + payload, zero-padded to ``width`` words."""
    w = [MAGIC, KINDS[kind], len(payload), version] + [int(v) & 0xFFFFFFFF for v in payload]
    assert len(w) <= width, (kind, len(w), width)
    return w + [0] * (width - len(w))


class ArgBlocks:
    """The engine program's arg blocks (one tensor per kind). ``add(kind, crs, per_device)`` where per_device is a list
    (one entry per mesh device, or a single entry replicated) of {(x, y): [payload u32...]} over the logical cores of
    ``crs``; cores of ``crs`` missing from a dict get an empty payload. Call before trace capture (allocates)."""

    def __init__(self, mesh):
        self.mesh = mesh
        self.blocks = {}  # kind -> dict(tensor, addr, width, cores)

    def add(self, kind, crs, per_device, version=1):
        import torch
        import ttnn

        assert kind in KINDS, kind
        cores = _cores(crs)
        ndev = self.mesh.get_num_devices()
        if len(per_device) == 1:
            per_device = per_device * ndev
        assert len(per_device) == ndev, (len(per_device), ndev)
        nmax = max((len(d.get((c.x, c.y), ())) for d in per_device for c in cores), default=0)
        width = -(-(HEADER + nmax) // 4) * 4
        shards = []
        for d in per_device:
            rows = [block_words(kind, d.get((c.x, c.y), ()), width, version) for c in cores]
            host = torch.tensor(rows, dtype=torch.int64).to(torch.int32).reshape(len(cores), width)  # wraps u32 bits
            shards.append(ttnn.from_torch(host, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT))
        mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                               ttnn.ShardSpec(crs, [1, width], ttnn.ShardOrientation.ROW_MAJOR))
        t = ttnn.to_device(ttnn.from_host_shards(shards, self.mesh.shape), self.mesh, memory_config=mc)
        self.blocks[kind] = dict(tensor=t, addr=t.buffer_address(), width=width, cores=cores)
        return self.blocks[kind]

    def common_args(self):
        """The common runtime args every engine kernel carries: [L1 address of kind 0, 1, ...] (0 = kind absent)."""
        out = [0] * len(KINDS)
        for k, b in self.blocks.items():
            out[KINDS[k]] = b["addr"]
        return out

    def tensors(self):
        """The block tensors, for the generic_op io list (keeps them alive / ordered with the program)."""
        return [self.blocks[k]["tensor"] for k in sorted(self.blocks, key=KINDS.get)]


def check_cb_ids(desc):
    """Raise if a ProgramDescriptor (or MeshProgramDescriptor values) uses a CB id > MAX_CB_ID (spec v1.25)."""
    progs = [desc] if hasattr(desc, "cbs") else list(desc.values()) if hasattr(desc, "values") else [desc]
    bad = sorted({f.buffer_index for p in progs for c in p.cbs for f in c.format_descriptors if f.buffer_index > MAX_CB_ID})
    if bad:
        raise ValueError(f"CB ids {bad} exceed {MAX_CB_ID}: the CB-config block is program-wide (spec v1.25)")
