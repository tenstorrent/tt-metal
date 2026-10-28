# SPDX-License-Identifier: Apache-2.0
"""Exact allocation-event ledger and alias census; no TTNN import at module load.

Capture setup in bounded graph windows. Feed each window to AllocationLedger,
then join the current tensor aliases. This records bank padding from the actual
allocator instead of estimating it from tensor shapes. It must run standalone,
never attached to a live serving server or inside a measured timing interval.
"""
from collections import defaultdict


class AllocationLedger:
    def __init__(self, banks):
        self.banks = banks
        self.live = {}
        self.peak = defaultdict(int)
        self.current = defaultdict(int)
        self.events = 0

    def update(self, graph):
        for node in graph:
            kind = node["node_type"]
            p = node.get("params", {})
            if kind not in ("buffer_allocate", "buffer_deallocate"):
                continue
            key = (int(p["device_id"]), p["type"], int(p["address"]))
            if kind == "buffer_allocate":
                if key in self.live:
                    raise ValueError(f"duplicate live allocation {key}")
                row = dict(p)
                row["physical_bytes_per_device"] = int(p["max_size_per_bank"]) * self.banks[p["type"]]
                self.live[key] = row
                domain = key[:2]
                self.current[domain] += row["physical_bytes_per_device"]
                self.peak[domain] = max(self.peak[domain], self.current[domain])
            else:
                if key not in self.live:
                    raise ValueError(f"pre-capture allocation freed: {key}")
                self.current[key[:2]] -= self.live.pop(key)["physical_bytes_per_device"]
            self.events += 1

    def snapshot(self, aliases, *, domain, allocator_bytes, baseline_bytes=0):
        """domain is the actual graph mesh allocator ID, not a chip ordinal.

        aliases contains path/address/buffer_type records from tensor_aliases.
        Each underlying allocation is counted once even if phase views or pooled
        states expose it under many Python attributes. Unmatched live allocations
        remain explicit; they are never silently labeled weights.
        """
        by_address = defaultdict(list)
        for alias in aliases:
            by_address[(alias["buffer_type"], alias["address"])].append(alias)
        rows = []
        for (device, kind, address), allocation in sorted(self.live.items()):
            if device != domain:
                continue
            rows.append(dict(allocation, aliases=by_address.get((kind, address), [])))
        dram = sum(r["physical_bytes_per_device"] for r in rows if r["type"] == "DRAM")
        expected = allocator_bytes - baseline_bytes
        if dram != expected:
            raise ValueError(f"allocator reconciliation failed: event DRAM={dram}, delta={expected}")
        matched = sum(r["physical_bytes_per_device"] for r in rows if r["type"] == "DRAM" and r["aliases"])
        return dict(
            allocations=rows,
            dram_bytes_per_device=dram,
            named_dram_bytes_per_device=matched,
            unnamed_dram_bytes_per_device=dram - matched,
            allocator_baseline_bytes_per_device=baseline_bytes,
            allocator_bytes_per_device=allocator_bytes,
            exact_allocator_reconciliation=True,
            event_count=self.events,
            event_dram_peak_bytes_per_device=baseline_bytes + self.peak[(domain, "DRAM")],
            scope="physical allocator bytes per device; mesh memory is not pooled; checkpoint is not a continuous peak",
        )


def tensor_aliases(roots):
    """Walk model-owned Python containers, recording physical root identity.

    Metadata calls only, with no tensor host readback. Runtime objects (devices,
    tokenizers, host checkpoint tensors) are excluded by the module filter.
    Repeated owner objects are listed separately to make shared pools visible.
    """
    import ttnn

    aliases, seen, object_aliases = [], {}, []

    def walk(value, path):
        if isinstance(value, ttnn.Tensor):
            if not value.is_allocated():
                return
            memory = value.memory_config()
            aliases.append(
                dict(
                    path=path,
                    address=value.buffer_address(),
                    root_buffer_unique_id=value.buffer_unique_id(),
                    buffer_type=str(memory.buffer_type).split(".")[-1],
                    shape=list(value.shape),
                    padded_shape=list(value.padded_shape),
                    dtype=str(value.dtype),
                    layout=str(value.layout),
                    buffer_page_size=value.buffer_page_size(),
                    buffer_num_pages=value.buffer_num_pages(),
                    buffer_aligned_page_size=value.buffer_aligned_page_size(),
                )
            )
            return
        if isinstance(value, (str, bytes, int, float, bool, type(None))):
            return
        if id(value) in seen:
            object_aliases.append(dict(path=path, owner=seen[id(value)]))
            return
        seen[id(value)] = path
        if isinstance(value, dict):
            for name, item in value.items():
                walk(item, f"{path}[{name!r}]")
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                walk(item, f"{path}[{i}]")
        elif type(value).__module__.startswith("models.") and hasattr(value, "__dict__"):
            for name, item in vars(value).items():
                walk(item, f"{path}.{name}")

    for name, value in roots.items():
        walk(value, name)
    return dict(tensors=aliases, shared_objects=object_aliases)
