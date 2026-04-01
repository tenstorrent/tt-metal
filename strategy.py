import math

from attr import dataclass
import logging
from typing import List

TILE_SIZE = 32


@dataclass(frozen=True)
class TSize:
    value: int

    @classmethod
    def from_tiles(cls, tiles: int) -> "TSize":
        return cls(tiles * TILE_SIZE)

    @property
    def tiles(self) -> int:
        return self.value // TILE_SIZE

    def add_tiles(self, tiles: int) -> "TSize":
        return TSize(self.value + tiles * TILE_SIZE)

    def __repr__(self) -> str:
        return f"TSize({self.value}, tiles={self.tiles})"

    def __str__(self) -> str:
        return f"{self.value} ({self.tiles}t)"


@dataclass
class ComputeGrid:
    cols: int = 11
    rows: int = 10

    @property
    def cores(self) -> int:
        return self.cols * self.rows


@dataclass
class LocalDeviceConfig:
    """Configuration for the strategy test."""

    NH: int = 32
    L: TSize = TSize(3200)
    grid: ComputeGrid = ComputeGrid()

    @property
    def cores(self) -> int:
        return self.grid.cores

    @property
    def total_tokens(self) -> TSize:
        return TSize(self.NH * self.L.value)


def evaluate_balance(c: LocalDeviceConfig, q_chunk_size: TSize):
    """Evaluate the balance of the strategy."""
    per_device_chunks = c.NH * (c.L.value / q_chunk_size.value) / c.cores
    logging.debug(f"Balance: {per_device_chunks:.2f} / {math.ceil(per_device_chunks)}")
    return per_device_chunks / math.ceil(per_device_chunks)


def get_tile_divisors(size: TSize) -> List[TSize]:
    """Return all TSize values that evenly divide the input in tiles."""
    divisors = []
    tiles = size.tiles
    for k in range(1, tiles + 1):
        if tiles % k == 0:
            divisors.append(TSize.from_tiles(k))
    return divisors


def get_q_candidates_that_divide(L: TSize) -> List[TSize]:
    """Get the list of q candidates based on L."""
    candidates = []
    k = 1
    candidate = k * TILE_SIZE
    while candidate <= L.value:
        if (L.value) % candidate == 0:
            candidates.append(TSize(candidate))
        k += 1
        candidate = k * TILE_SIZE
    return candidates


def get_candidates_relaxed(cfg: LocalDeviceConfig) -> List[TSize]:
    """Get the list of q candidates based on L, allowing for some relaxation."""
    total_tokens = cfg.total_tokens
    tokens_per_core_float = total_tokens.value / cfg.cores
    logging.info(f"Total tokens: {total_tokens}, Tokens per core (perfect): {tokens_per_core_float:.2f}")

    tokens_per_core = TSize(math.ceil(tokens_per_core_float / TILE_SIZE) * TILE_SIZE)
    tokens_per_core_to_try = [tokens_per_core]
    while tokens_per_core_float / (tokens_per_core.add_tiles(1).value) >= 0.88:
        tokens_per_core = tokens_per_core.add_tiles(1)
        tokens_per_core_to_try.append(tokens_per_core)

    candidates = []
    for tokens in tokens_per_core_to_try:
        q_candidates = get_tile_divisors(tokens)
        candidates.extend(q_candidates)
    candidates = list(set(candidates))  # Remove duplicates
    logging.info(f"Relaxed candidates: {candidates}")
    return candidates


def brute_force_candidates(cfg: LocalDeviceConfig) -> List[TSize]:
    return [TSize.from_tiles(x) for x in range(1, 20) if x <= cfg.L.tiles]


@dataclass
class CoreCoord:
    """A core coordinate in the grid."""

    x: int
    y: int

    def __str__(self) -> str:
        return f"({self.y},{self.x})"

    def __repr__(self) -> str:
        return self.__str__()


def linear_to_zigzag(linear_idx: int, num_chunks: int) -> int:
    """Convert linear index to zigzag index for load balancing.

    Zigzag pairs light work (early chunks) with heavy work (late chunks):
    - Position 0 -> chunk 0 (lightest)
    - Position 1 -> chunk n-1 (heaviest)
    - Position 2 -> chunk 1
    - Position 3 -> chunk n-2
    """
    if linear_idx % 2 == 0:
        return linear_idx // 2
    else:
        return num_chunks - 1 - (linear_idx // 2)


@dataclass
class CoreHeadWork:
    """Work assignment for a single head on a core."""

    head: int
    q_chunk_start: int
    q_chunk_count: int

    def __str__(self) -> str:
        indices = range(self.q_chunk_start, self.q_chunk_start + self.q_chunk_count)
        return f"H:{self.head}-Q:[{','.join(str(i) for i in indices)}]"

    def __repr__(self) -> str:
        return self.__str__()

    def to_zigzag_str(self, num_q_chunks: int) -> str:
        """Return string with zigzag-converted indices and L/H counts."""
        indices = range(self.q_chunk_start, self.q_chunk_start + self.q_chunk_count)
        zigzag_indices = [linear_to_zigzag(i, num_q_chunks) for i in indices]
        half = num_q_chunks // 2
        light_cnt = sum(1 for i in zigzag_indices if i < half)
        heavy_cnt = len(zigzag_indices) - light_cnt
        return f"H:{self.head}-Q:[{','.join(str(i) for i in zigzag_indices)}]{{L{light_cnt}+H{heavy_cnt}}}"


@dataclass
class CoreWork:
    """All work assigned to a single core."""

    coord: CoreCoord
    global_q_start: int
    global_q_count: int
    head_work: List["CoreHeadWork"]


@dataclass
class Chain:
    """A chain of consecutive cores processing a single head (L).

    A chain represents cores that need to communicate/synchronize when
    processing chunks of the same head. Cores are ordered by their position
    in the processing sequence.
    """

    head: int
    cores: List[CoreCoord]

    def __str__(self) -> str:
        core_str = "->".join(str(c) for c in self.cores)
        return f"Chain(head={self.head}: {core_str})"

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class ChainBreak:
    """Configuration for chain breaking conditions."""

    no_multi_row: bool = False  # Break chains that span multiple rows
    no_multi_head_core: bool = False  # Break chains at cores that process two heads


@dataclass
class ChainWorkload:
    """Chain with associated workload information for printing."""

    chain: Chain
    core_work: List[CoreWork]  # CoreWork for each core in the chain
    head_work_indices: List[int]  # Index into core_work[i].head_work for this chain's head

    def get_chunk_counts(self) -> List[int]:
        """Get chunk count for each core in the chain."""
        return [
            self.core_work[i].head_work[self.head_work_indices[i]].q_chunk_count for i in range(len(self.chain.cores))
        ]

    def max_workload(self) -> int:
        """Maximum workload of any core in the chain."""
        chunks = self.get_chunk_counts()
        return max(chunks) if chunks else 0

    def total_workload(self) -> int:
        """Total workload across all cores in the chain."""
        return sum(self.get_chunk_counts())

    def dram_read_ratio(self) -> float:
        """Ratio of DRAM reads: max_workload / total_workload.

        Uses max because a core may process multiple chains, so the bottleneck
        is the core with the most work in this chain.
        Lower is better - means more data is passed via L1 instead of DRAM.
        """
        total = self.total_workload()
        return self.max_workload() / total if total > 0 else 1.0

    def __str__(self) -> str:
        chunks = self.get_chunk_counts()
        parts = [f"{coord}[{c}]" for coord, c in zip(self.chain.cores, chunks)]
        ratio = self.dram_read_ratio()
        return f"Chain(head={self.chain.head}: {'->'.join(parts)}, dram_ratio={ratio:.2f})"

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class ConstructChainsResult:
    """Result of construct_chains containing chains and work distribution."""

    chains: List[Chain]
    core_work: List[CoreWork]
    head_segments: List[List[tuple]]  # head -> list of (core_idx, head_work_idx)


def construct_chains(cfg: LocalDeviceConfig, q_chunk_size: TSize, enable_zigzag: bool) -> ConstructChainsResult:
    """
    Construct a list of chains given q_chunk_size and LocalDeviceConfig.

    A chain is a list of consecutive cores in a grid that process a single L
    (there are NH of them). One core can process up to two heads.

    Work distribution:
    - Total chunks = NH * (L / q_chunk_size)
    - Distributed round-robin: first cores get ceil(chunks/cores),
      last cores get floor(chunks/cores)
    - When enable_zigzag=True, chunks are distributed in pairs to ensure
      each core gets balanced light+heavy work

    Args:
        cfg: LocalDeviceConfig with NH, L, and grid settings
        q_chunk_size: Size of Q chunks
        enable_zigzag: If True, distribute chunks in pairs for zigzag balancing

    Returns:
        ConstructChainsResult containing chains, core_work, and head_segments
    """
    num_cores = cfg.cores
    grid = cfg.grid

    # Total chunks to process
    num_q_chunks_per_head = cfg.L.value // q_chunk_size.value
    total_q_chunks = cfg.NH * num_q_chunks_per_head

    # Distribute chunks across cores (round-robin)
    # When zigzag is enabled, distribute pairs to ensure balanced light+heavy work
    if enable_zigzag and num_q_chunks_per_head % 2 == 0:
        logging.info("Using zigzag distribution of chunks to balance light/heavy work")
        total_pairs = total_q_chunks // 2
        cores_doing_extra_work = total_pairs % num_cores if num_cores > 0 else 0
        base_chunks_per_core = (total_pairs // num_cores) * 2 if num_cores > 0 else 0
        extra_chunks_per_core = 2
    else:
        logging.info("Using standard distribution of chunks (without zigzag)")
        cores_doing_extra_work = total_q_chunks % num_cores if num_cores > 0 else 0
        base_chunks_per_core = total_q_chunks // num_cores if num_cores > 0 else 0
        extra_chunks_per_core = 1

    logging.info(
        f"Total Q chunks: {total_q_chunks}, Chunks/core: {base_chunks_per_core} (+{extra_chunks_per_core} for first {cores_doing_extra_work} cores)"
    )

    # Build core work assignments
    core_work: List[CoreWork] = []
    next_global_chunk = 0

    def decode_flat_chunk(flat_chunk_index: int) -> tuple:
        """Decode flat chunk index into (head, q_chunk_within_head)."""
        if num_q_chunks_per_head == 0:
            return (0, 0)
        head = flat_chunk_index // num_q_chunks_per_head
        q_chunk = flat_chunk_index % num_q_chunks_per_head
        return (head, q_chunk)

    for i in range(num_cores):
        # Core coordinate: x varies first, then y
        coord = CoreCoord(x=i % grid.cols, y=i // grid.cols)

        # This core gets ceil or floor chunks based on position
        chunk_count = base_chunks_per_core + (extra_chunks_per_core if i < cores_doing_extra_work else 0)
        chunk_count = min(chunk_count, total_q_chunks - next_global_chunk)
        chunk_count = max(chunk_count, 0)

        head_work: List[CoreHeadWork] = []
        remaining = chunk_count
        flat_chunk = next_global_chunk

        while remaining > 0:
            head, q_chunk_idx = decode_flat_chunk(flat_chunk)
            # How many chunks can we take from this head?
            chunks_left_in_head = num_q_chunks_per_head - q_chunk_idx
            chunk_take = min(remaining, chunks_left_in_head)

            head_work.append(CoreHeadWork(head=head, q_chunk_start=q_chunk_idx, q_chunk_count=chunk_take))

            remaining -= chunk_take
            flat_chunk += chunk_take

        core_work.append(
            CoreWork(coord=coord, global_q_start=next_global_chunk, global_q_count=chunk_count, head_work=head_work)
        )

        next_global_chunk += chunk_count

    # Build head -> list of (core_idx, head_work_idx) mapping
    head_segments: List[List[tuple]] = [[] for _ in range(cfg.NH)]
    for core_idx, work in enumerate(core_work):
        for hw_idx, hw in enumerate(work.head_work):
            if hw.head < cfg.NH:
                head_segments[hw.head].append((core_idx, hw_idx))

    # Construct chains: each head that spans >= 1 core becomes a chain
    chains: List[Chain] = []
    for head in range(cfg.NH):
        segments = head_segments[head]
        if len(segments) == 0:
            continue

        # Collect cores for this head in order
        core_coords = [core_work[seg[0]].coord for seg in segments]
        chains.append(Chain(head=head, cores=core_coords))

    return ConstructChainsResult(chains=chains, core_work=core_work, head_segments=head_segments)


def _build_core_to_segment_lookup(result: ConstructChainsResult) -> dict:
    """Build lookup: (core_x, core_y, head) -> segment."""
    lookup = {}
    for head_idx, segments in enumerate(result.head_segments):
        for seg in segments:
            core = result.core_work[seg[0]].coord
            lookup[(core.x, core.y, head_idx)] = seg
    return lookup


def _get_chain_segments(chain: Chain, lookup: dict) -> List[tuple]:
    """Get segments for a chain's cores using the lookup."""
    return [lookup[(c.x, c.y, chain.head)] for c in chain.cores]


def break_chains_by_row(result: ConstructChainsResult, break_config: ChainBreak) -> ConstructChainsResult:
    """Break chains that span multiple rows."""
    if not break_config.no_multi_row:
        return result

    lookup = _build_core_to_segment_lookup(result)
    new_chains: List[Chain] = []
    new_head_segments: List[List[tuple]] = [[] for _ in range(len(result.head_segments))]

    for chain in result.chains:
        segments = _get_chain_segments(chain, lookup)
        if len(segments) <= 1:
            new_chains.append(chain)
            new_head_segments[chain.head].extend(segments)
            continue

        current_row = chain.cores[0].y
        current_chain_cores: List[CoreCoord] = []
        current_chain_segments: List[tuple] = []

        for coord, seg in zip(chain.cores, segments):
            if coord.y != current_row and current_chain_cores:
                new_chains.append(Chain(head=chain.head, cores=current_chain_cores))
                new_head_segments[chain.head].extend(current_chain_segments)
                current_chain_cores = []
                current_chain_segments = []
                current_row = coord.y

            current_chain_cores.append(coord)
            current_chain_segments.append(seg)

        if current_chain_cores:
            new_chains.append(Chain(head=chain.head, cores=current_chain_cores))
            new_head_segments[chain.head].extend(current_chain_segments)

    return ConstructChainsResult(chains=new_chains, core_work=result.core_work, head_segments=new_head_segments)


def break_chains_by_multi_head(result: ConstructChainsResult, break_config: ChainBreak) -> ConstructChainsResult:
    """Break chains at cores that process multiple heads."""
    if not break_config.no_multi_head_core:
        return result

    lookup = _build_core_to_segment_lookup(result)
    new_chains: List[Chain] = []
    new_head_segments: List[List[tuple]] = [[] for _ in range(len(result.head_segments))]

    for chain in result.chains:
        segments = _get_chain_segments(chain, lookup)
        if len(segments) <= 1:
            new_chains.append(chain)
            new_head_segments[chain.head].extend(segments)
            continue

        current_chain_cores: List[CoreCoord] = []
        current_chain_segments: List[tuple] = []

        for coord, seg in zip(chain.cores, segments):
            core_idx = seg[0]
            is_multi_head = len(result.core_work[core_idx].head_work) > 1

            if is_multi_head:
                if current_chain_cores:
                    new_chains.append(Chain(head=chain.head, cores=current_chain_cores))
                    new_head_segments[chain.head].extend(current_chain_segments)
                    current_chain_cores = []
                    current_chain_segments = []
                # Multi-head core becomes its own single-core chain
                new_chains.append(Chain(head=chain.head, cores=[coord]))
                new_head_segments[chain.head].append(seg)
            else:
                current_chain_cores.append(coord)
                current_chain_segments.append(seg)

        if current_chain_cores:
            new_chains.append(Chain(head=chain.head, cores=current_chain_cores))
            new_head_segments[chain.head].extend(current_chain_segments)

    return ConstructChainsResult(chains=new_chains, core_work=result.core_work, head_segments=new_head_segments)


def break_chains(result: ConstructChainsResult, break_config: ChainBreak) -> ConstructChainsResult:
    """Break chains based on ChainBreak configuration."""
    result = break_chains_by_row(result, break_config)
    result = break_chains_by_multi_head(result, break_config)
    return result


def build_chain_workloads(result: ConstructChainsResult) -> List[ChainWorkload]:
    """Build ChainWorkload objects from construct_chains result for printing."""
    workloads = []
    for chain in result.chains:
        # Find the segments for this specific chain (by matching cores)
        all_segments = result.head_segments[chain.head]
        # Filter segments to only those whose cores match this chain
        chain_segments = []
        core_set = set((c.x, c.y) for c in chain.cores)
        for seg in all_segments:
            core = result.core_work[seg[0]].coord
            if (core.x, core.y) in core_set:
                chain_segments.append(seg)

        chain_core_work = [result.core_work[seg[0]] for seg in chain_segments]
        hw_indices = [seg[1] for seg in chain_segments]
        workloads.append(ChainWorkload(chain=chain, core_work=chain_core_work, head_work_indices=hw_indices))
    return workloads


def calculate_chain_dram_savings(
    cfg: LocalDeviceConfig, q_chunk_size: TSize, break_config: ChainBreak = None, enable_zigzag: bool = True
) -> float:
    """Calculate DRAM savings based on chain construction and breaking."""
    if break_config is None:
        break_config = ChainBreak()

    num_cores = cfg.cores
    grid = cfg.grid

    num_q_chunks_per_head = cfg.L.value // q_chunk_size.value
    total_q_chunks = cfg.NH * num_q_chunks_per_head

    # Match zigzag pair distribution logic from C++
    if enable_zigzag and num_q_chunks_per_head % 2 == 0:
        total_pairs = total_q_chunks // 2
        cores_doing_extra_work = total_pairs % num_cores if num_cores > 0 else 0
        base_chunks_per_core = (total_pairs // num_cores) * 2 if num_cores > 0 else 0
        extra_chunks_per_core = 2
    else:
        cores_doing_extra_work = total_q_chunks % num_cores if num_cores > 0 else 0
        base_chunks_per_core = total_q_chunks // num_cores if num_cores > 0 else 0
        extra_chunks_per_core = 1

    logging.debug(f"{'='*60}")
    logging.debug(f"Work Distribution (zigzag={enable_zigzag})")
    logging.debug(f"{'='*60}")
    logging.debug(f"Config: NH={cfg.NH}, L={cfg.L}, grid={grid.cols}x{grid.rows}")
    logging.debug(f"q_chunk_size: {q_chunk_size}")
    logging.debug(f"Chunks per head: {num_q_chunks_per_head}")
    logging.debug(f"Total chunks: {total_q_chunks}")
    logging.debug(f"Cores: {num_cores}")
    logging.debug(f"Base chunks/core: {base_chunks_per_core}, extra_per_core: {extra_chunks_per_core}")
    logging.debug(f"First {cores_doing_extra_work} cores get {base_chunks_per_core + extra_chunks_per_core} chunks")
    logging.debug(f"Remaining {num_cores - cores_doing_extra_work} cores get {base_chunks_per_core} chunks")
    logging.debug(f"Chain break config: {break_config}")

    result = construct_chains(cfg, q_chunk_size, enable_zigzag=enable_zigzag)
    result = break_chains(result, break_config)

    # Log per-core workload in H:<head>Q:[zigzag indices] format
    logging.debug(f"{'='*60}")
    logging.debug(f"Per-Core Workload (zigzag indices)")
    logging.debug(f"{'='*60}")
    for i, work in enumerate(result.core_work):
        if work.global_q_count == 0:
            continue
        logging.debug(
            f"Core {i} {work.coord}: {' '.join(hw.to_zigzag_str(num_q_chunks_per_head) for hw in work.head_work)}"
        )

    chain_workloads = build_chain_workloads(result)

    logging.debug(f"{'='*60}")
    logging.debug(f"Chains ({len(chain_workloads)} total)")
    logging.debug(f"{'='*60}")
    for cw in chain_workloads:
        logging.debug(cw)

    # Compute total DRAM access ratio across all chains
    total_max_workload = sum(cw.max_workload() for cw in chain_workloads)
    total_workload = sum(cw.total_workload() for cw in chain_workloads)
    total_dram_ratio = total_max_workload / total_workload if total_workload > 0 else 1.0
    logging.debug(f"{'='*60}")
    logging.debug(f"Total DRAM access ratio: {total_dram_ratio:.2f} ({total_max_workload}/{total_workload})")
    return total_dram_ratio


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname).1s]: %(message)s", datefmt="%H:%M:%S")
    cfg = LocalDeviceConfig(NH=32, L=TSize(3200), grid=ComputeGrid(cols=11, rows=10))
    q_candidates = get_candidates_relaxed(cfg)
    q_candidates = brute_force_candidates(cfg)

    logging.debug(f"q candidates: {q_candidates}")
    balances = [(q, evaluate_balance(cfg, q)) for q in q_candidates]
    balances_sorted = sorted(balances, key=lambda x: x[1], reverse=True)
    for q, bal in balances_sorted:
        logging.info(f"q_chunk_size: {q}, Balance: {bal:.2f}")

    # qs = [b[0] for b in balances_sorted if b[1] > 0.90]
    qs = [TSize(160)]

    # Collect results for summary table
    results = []
    for q in qs:
        logging.debug(f"\nCalculating Chain DRAM savings for q_chunk_size: {q}")
        mh_mr = calculate_chain_dram_savings(cfg, q)  # multihead OK, multirow OK
        mh_xr = calculate_chain_dram_savings(cfg, q, ChainBreak(no_multi_row=True))  # multihead OK, no multirow
        xh_mr = calculate_chain_dram_savings(cfg, q, ChainBreak(no_multi_head_core=True))  # no multihead, multirow OK
        xh_xr = calculate_chain_dram_savings(
            cfg, q, ChainBreak(no_multi_row=True, no_multi_head_core=True)
        )  # no multihead, no multirow
        results.append((q, mh_mr, mh_xr, xh_mr, xh_xr))

    # Print summary table
    # Headers: [Multihead][Multirow] with check/uncheck
    logging.info("")
    logging.info("=" * 80)
    logging.info("DRAM Access Ratio Summary (lower is better)")
    logging.info("=" * 80)
    logging.info(f"{'Q_CHUNK':<12} {'[MH:+][MR:+]':<14} {'[MH:+][MR:-]':<14} {'[MH:-][MR:+]':<14} {'[MH:-][MR:-]':<14}")
    logging.info("-" * 80)
    for q, mh_mr, mh_xr, xh_mr, xh_xr in results:
        logging.info(f"{str(q):<12} {mh_mr:<14.2f} {mh_xr:<14.2f} {xh_mr:<14.2f} {xh_xr:<14.2f}")
