"""Import-light 2K migration edge-case oracle; this module never opens a device.

Page coordinates are logical (config, slot, layer, global token start). Native KVM
copies whole intersecting pages. Bytes outside a selected token interval but inside
its boundary page are therefore part of the expected transfer, not sentinels.
"""

from dataclasses import dataclass
from hashlib import sha256

CAPACITY = 2048
PAGE_TOKENS = 32
PAGE_BYTES = 4352
CONFIGS = tuple(f"{kind}_h{head}" for kind in ("k", "v") for head in range(8))


def integer(value, name, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
    return value


def pages(begin, end):
    """Return every complete 32-token page intersecting a nonempty token range."""
    integer(begin, "begin", 0, CAPACITY - 1)
    integer(end, "end", 1, CAPACITY)
    if begin >= end:
        raise ValueError("range must be nonempty")
    return range(begin // PAGE_TOKENS * PAGE_TOKENS, (end + PAGE_TOKENS - 1) // PAGE_TOKENS * PAGE_TOKENS, PAGE_TOKENS)


@dataclass(frozen=True)
class Call:
    name: str
    prompt: str
    slot: int
    begin: int
    end: int

    def __post_init__(self):
        integer(self.slot, "slot", 0, 1)
        pages(self.begin, self.end)
        if self.begin % PAGE_TOKENS or self.end - self.begin > 1024:
            raise ValueError("runtime chunk must start on a tile and contain at most 1024 tokens")


# C is a third, distinct token fixture. C-extend retains exactly C's first 33 tokens.
# The owner must prove previous native reads drained before admitting C to slot 0.
RUNTIME_CALLS = (
    Call("A-first", "A", 0, 0, 1024),
    Call("B-sp-tail", "B", 1, 0, 257),
    Call("A-chunk-tail", "A", 0, 1024, 1033),
    Call("C-reuses-slot0", "C", 0, 0, 33),
    Call("C-tile-continuation", "C", 0, 32, 65),
)

# Independent layers let all nine writes share one before/after snapshot pair.
WRITER_CASES = tuple(
    zip(
        (0, 1, 7, 8, 15, 16, 23, 24, 31),
        (0, 1, 0, 1, 0, 1, 0, 1, 0),
        ((0, 31), (0, 32), (0, 33), (224, 255), (224, 256), (224, 257), (992, 1023), (992, 1024), (992, 1025)),
    )
)

# Each is a distinct transfer generation with the same retained A source content.
# Adjacent bursts intentionally overlap a partial page; final coverage is [0,1056).
PREFIX_RANGES = ((0, 33), (32, 257), (256, 1033))


def token_regions(call):
    """Keep semantic valid length separate from the complete packed page footprint."""
    rounded = (call.end + 31) // 32 * 32
    return dict(
        preserved_prefix=(0, call.begin),
        valid=(call.begin, call.end),
        zero_padding=(call.end, rounded),
        preserved_suffix=(rounded, CAPACITY),
    )


def keys():
    for config in range(16):
        for slot in range(2):
            for layer in range(32):
                for position in range(0, CAPACITY, PAGE_TOKENS):
                    yield (config, slot, layer, position)


def sp_address(position):
    """Independent block-cyclic SP coordinate and chip-local row (no table lookup)."""
    integer(position, "position", 0, CAPACITY - 1)
    return (position // 256) % 4, (position // 1024) * 256 + position % 256


def h2d_reference_rows(token_ids, call, *, pad_id=0):
    """Independent oracle only; the live owner must use production pack_token_ids."""
    if len(token_ids) != call.end - call.begin:
        raise ValueError("tokens must contain exactly the valid interval")
    for token in (*token_ids, pad_id):
        integer(token, "token", 0, 128255)
    rows = [[] for _ in range(4)]
    for offset in range(1024):
        row = ((call.begin + offset) // 256) % 4
        rows[row].append(token_ids[offset] if offset < len(token_ids) else pad_id)
    if any(len(row) != 256 for row in rows):
        raise AssertionError("SP row must contain one physical 256-token block")
    return rows


def prepare_h2d_input(token_ids, call, pack_token_ids):
    """Replace the passing gate's full-length-only reshape with its production packer.

    The caller converts the returned host tensor to uint32 and reshapes [4,1,256]
    for the persistent H2D service. Borrowed device tensors still go directly to
    runtime.prefill_chunk; this helper must never receive such a tensor.
    """
    if not isinstance(token_ids, (list, tuple)):
        raise TypeError("expected owned host token IDs, not a borrowed device tensor")
    h2d_reference_rows(token_ids, call)
    return pack_token_ids(token_ids, actual_start=call.begin, actual_end=call.end, pad_id=0, max_seq_len=CAPACITY)


def _read(reader, key):
    value = reader(key)
    if not isinstance(value, bytes) or len(value) != PAGE_BYTES:
        raise ValueError(f"invalid packed page at {key}")
    return value


def verify_transfer(before, after, source, *, source_slot, destination_slot, begin, end):
    """Stream all 65536 destination pages against source bytes or untouched pre-state.

    Readers must bind immutable snapshot identity externally. Reading a live source
    while a reused slot is changing would invalidate the comparison. This function
    is an oracle, not a client, manager, cleanup supervisor, or device adapter.
    """
    integer(source_slot, "source_slot", 0, 1)
    integer(destination_slot, "destination_slot", 0, 1)
    selected = set(pages(begin, end))
    count = changed = mismatches = 0
    examples = []
    digest = sha256()
    for key in keys():
        config, slot, layer, position = key
        old = _read(before, key)
        if slot == destination_slot and position in selected:
            expected = _read(source, (config, source_slot, layer, position))
            changed += 1
        else:
            expected = old
        actual = _read(after, key)
        digest.update(actual)
        count += 1
        if actual != expected:
            mismatches += 1
            if len(examples) < 8:
                examples.append(key)
    return dict(
        pages=count,
        selected_pages=changed,
        untouched_pages=count - changed,
        selected_bytes=changed * PAGE_BYTES,
        mismatches=mismatches,
        examples=examples,
        after_sha256=digest.hexdigest(),
        passed=mismatches == 0,
    )


def require_distinct_prompts(fixtures):
    if set(fixtures) != {"A", "B", "C"}:
        raise ValueError("exactly A/B/C fixtures are required")
    for name, minimum in (("A", 1033), ("B", 257), ("C", 65)):
        if not isinstance(fixtures[name], (list, tuple)) or len(fixtures[name]) < minimum:
            raise ValueError(f"short fixture {name}")
        for token in fixtures[name]:
            integer(token, "token", 0, 128255)
    # Compare equal-length prefixes; length differences alone do not prove a third prompt.
    prefixes = {tuple(fixtures[name][:33]) for name in ("A", "B", "C")}
    if len(prefixes) != 3:
        raise ValueError("A/B/C must have distinct first 33 token IDs")
