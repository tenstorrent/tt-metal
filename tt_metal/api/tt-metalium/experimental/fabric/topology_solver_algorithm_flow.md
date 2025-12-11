# Topology Solver Algorithm Flow

## High-Level Algorithm Flow

```
┌─────────────────────────────────────────────────────────────┐
│ solve_topology_mapping(target_graph, global_graph, constraints) │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │ Stage 1: Preprocessing                 │
        │ - Build GraphIndexData                 │
        │ - Build ConstraintIndexData            │
        │ - Validate constraints                 │
        └───────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │ Stage 2: Fast Path Detection          │
        │ - Check if target is path graph       │
        │ - If yes: try path-extension DFS      │
        │ - If succeeds: return result          │
        └───────────────────────────────────────┘
                            │
                            ▼ (if fast path fails)
        ┌───────────────────────────────────────┐
        │ Stage 3: General DFS Search            │
        │ - Initialize search state             │
        │ - Handle pinned nodes                  │
        │ - Try multiple starting points         │
        │ - Run DFS with backtracking           │
        └───────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │ Stage 4: Validation                   │
        │ - Validate mapping                    │
        │ - Strict mode checks                  │
        │ - Build result                        │
        └───────────────────────────────────────┘
                            │
                            ▼
                    Return MappingResult
```

## Detailed DFS Flow

```
DFS(pos, mapping, used, ...)
│
├─► Base Case: pos == n_target
│   └─► return true (success)
│
├─► Check memoization cache
│   └─► if state already failed: return false
│
├─► Select next target node (MRV heuristic)
│   └─► Prefer: most mapped neighbors → fewest candidates → lowest degree
│
├─► Generate candidates for target node
│   ├─► Filter by constraints
│   ├─► Filter by degree
│   ├─► Filter by local consistency
│   └─► Order by degree gap (tighter fits first)
│
├─► For each candidate:
│   │
│   ├─► Check local consistency
│   │   ├─► Mapped neighbors must be connected
│   │   └─► Strict mode: check channel counts
│   │
│   ├─► Forward checking
│   │   ├─► Count unused neighbors
│   │   ├─► Verify future neighbors have candidates
│   │   └─► Check reachability (for path graphs)
│   │
│   ├─► Assign: mapping[target] = candidate, used[candidate] = true
│   │
│   ├─► Recursive call: DFS(pos + 1, ...)
│   │   └─► if succeeds: return true
│   │
│   └─► Backtrack: mapping[target] = -1, used[candidate] = false
│
└─► Mark state as failed in memoization cache
    └─► return false
```

## Data Flow

```
Input:
├─ AdjacencyGraph<TargetNode> target_graph
├─ AdjacencyGraph<GlobalNode> global_graph
└─ MappingConstraints<TargetNode, GlobalNode> constraints

    │
    ▼

Preprocessing:
├─ GraphIndexData
│  ├─ target_nodes: vector<TargetNode>
│  ├─ global_nodes: vector<GlobalNode>
│  ├─ target_adj_idx: vector<vector<size_t>>
│  ├─ global_adj_idx: vector<vector<size_t>>
│  ├─ target_deg: vector<size_t>
│  └─ global_deg: vector<size_t>
│
└─ ConstraintIndexData
   ├─ restricted_global_indices: vector<vector<size_t>>
   └─ preferred_global_indices: vector<vector<size_t>>

    │
    ▼

Search State:
├─ mapping: vector<int>  (mapping[target_idx] = global_idx or -1)
├─ used: vector<bool>    (used[global_idx] = true if assigned)
└─ failed_states: unordered_set<uint64_t>

    │
    ▼

Output:
└─ MappingResult
   ├─ success: bool
   ├─ target_to_global: map<TargetNode, GlobalNode>
   ├─ global_to_target: map<GlobalNode, TargetNode>
   └─ stats: {dfs_calls, backtrack_count, elapsed_time}
```

## Key Algorithms

### 1. Fast Path Detection

```
is_path_graph(graph):
  endpoints = []
  for each node in graph:
    if degree == 1:
      endpoints.append(node)
    if degree > 2:
      return false
  return endpoints.size() == 2
```

### 2. MRV Node Selection

```
select_next_target(mapping, used):
  best = None
  best_mapped_neighbors = 0
  best_candidate_count = INF
  best_degree = INF

  for each unassigned target node:
    mapped_neighbors = count(mapped neighbors)
    candidate_count = count_valid_candidates(node)

    if mapped_neighbors > best_mapped_neighbors OR
       (mapped_neighbors == best_mapped_neighbors AND
        candidate_count < best_candidate_count) OR
       (mapped_neighbors == 0 AND degree < best_degree):
      best = node
      update best_* values

  return best
```

### 3. Candidate Generation

```
generate_candidates(target_idx, mapping, used):
  candidates = []

  // Get constraint-restricted candidates
  if has_constraints(target_idx):
    candidates = constraint_candidates[target_idx]
  else:
    candidates = all_global_nodes

  // Filter candidates
  filtered = []
  for candidate in candidates:
    if used[candidate]:
      continue
    if global_deg[candidate] < target_deg[target_idx]:
      continue
    if not is_valid_mapping(target_idx, candidate):
      continue
    if not check_local_consistency(target_idx, candidate, mapping):
      continue
    if not check_forward_consistency(target_idx, candidate, mapping, used):
      continue
    filtered.append(candidate)

  // Order by degree gap
  sort(filtered, by: (global_deg - target_deg))

  return filtered
```

### 4. Consistency Checking

```
check_local_consistency(target_idx, global_idx, mapping):
  for each mapped neighbor of target_idx:
    neighbor_global = mapping[neighbor]
    if not connected(global_idx, neighbor_global):
      return false
    if strict_mode:
      if channel_count(global_idx, neighbor_global) <
         required_count(target_idx, neighbor):
        return false
  return true

check_forward_consistency(target_idx, global_idx, mapping, used):
  unassigned_neighbors = unmapped neighbors of target_idx
  unused_global_neighbors = unused neighbors of global_idx

  if unused_global_neighbors.size() < unassigned_neighbors.size():
    return false

  for each unassigned_neighbor:
    has_candidate = false
    for each unused_global_neighbor:
      if can_map(unassigned_neighbor, unused_global_neighbor, mapping):
        has_candidate = true
        break
    if not has_candidate:
      return false

  return true
```

## Optimization Techniques

### 1. Memoization
- Hash partial assignments to avoid revisiting failed states
- Uses FNV-1a hash function
- Stores hash of (target_idx, global_idx) pairs

### 2. Forward Checking
- Prune candidates that would leave future nodes without options
- Count unused neighbors vs. unassigned neighbors
- Verify each future neighbor has at least one viable candidate

### 3. Candidate Ordering
- Order by degree gap: prefer tighter fits
- Reduces search space by trying promising candidates first

### 4. Early Termination
- Fast path for path graphs: O(n) instead of exponential
- Memoization: avoid redundant work
- Forward checking: prune early

## Complexity Analysis

### Time Complexity
- **Preprocessing**: O(n + m + E_target + E_global)
  - n = target nodes, m = global nodes
  - E_target = target edges, E_global = global edges

- **Fast Path**: O(n * m) worst case, typically O(n)
  - Linear path extension with neighbor checks

- **General DFS**: O(m^n) worst case, but heavily pruned
  - Exponential in worst case
  - Typically much better due to pruning
  - Memoization reduces redundant work

### Space Complexity
- **GraphIndexData**: O(n + m + E_target + E_global)
- **ConstraintIndexData**: O(n * m) worst case
- **Search State**: O(n + m + failed_states)
- **Total**: O(n + m + E + failed_states)

## Example: Mapping a 3-Node Path

```
Target Graph:     Global Graph:
  1 -- 2 -- 3        10 -- 11 -- 12 -- 13

Step 1: Select node 1 (no mapped neighbors, degree 1)
  Candidates: {10, 11, 12, 13} (all have degree >= 1)
  Try: 1 → 10

Step 2: Select node 2 (1 mapped neighbor, degree 2)
  Candidates: {11} (only neighbor of 10, degree >= 2)
  Try: 2 → 11

Step 3: Select node 3 (1 mapped neighbor, degree 1)
  Candidates: {12} (neighbor of 11, degree >= 1)
  Try: 3 → 12

Success! Mapping: {1→10, 2→11, 3→12}
```

## Example: Mapping with Constraints

```
Target Graph:     Global Graph:
  1 -- 2           10 -- 11
                   12 -- 13

Constraints:
  - Node 1 must map to host 0 (10, 11)
  - Node 2 must map to host 1 (12, 13)

Step 1: Select node 1
  Candidates: {10, 11} (filtered by constraint)
  Try: 1 → 10

Step 2: Select node 2
  Candidates: {12, 13} (filtered by constraint)
  But: 10 not connected to {12, 13}
  Backtrack!

Step 1 (retry): Select node 1
  Candidates: {10, 11}
  Try: 1 → 11

Step 2 (retry): Select node 2
  Candidates: {12, 13}
  But: 11 not connected to {12, 13}
  Backtrack!

Failure: No valid mapping exists
```
