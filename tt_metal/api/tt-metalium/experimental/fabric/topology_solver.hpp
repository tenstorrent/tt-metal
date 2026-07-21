// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <climits>
#include <cstddef>
#include <algorithm>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph_descriptor.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>

namespace tt::tt_metal {
class PhysicalSystemDescriptor;
}  // namespace tt::tt_metal

namespace tt::tt_fabric {

/**
 * @brief Generic graph representation with minimal query interface
 *
 * AdjacencyGraph provides a generic graph representation that works with any node type.
 * It provides a minimal interface for querying graph structure: getting all nodes and
 * getting neighbors of a specific node.
 *
 * @tparam NodeId The type used to identify nodes in the graph
 */
template <typename NodeId>
class AdjacencyGraph {
public:
    using NodeType = NodeId;
    using AdjacencyMap = std::map<NodeId, std::vector<NodeId>>;

    /**
     * @brief Construct empty adjacency graph
     */
    AdjacencyGraph() = default;

    /**
     * @brief Construct adjacency graph from a MeshGraph
     *
     * @param mesh_graph The mesh graph to construct the adjacency graph from
     */
    explicit AdjacencyGraph(const AdjacencyMap& adjacency_map);

    /**
     * @brief Get all nodes in the graph
     *
     * @return const std::vector<NodeId>& Vector of all node IDs in the graph
     */
    const std::vector<NodeId>& get_nodes() const;

    /**
     * @brief Get neighbors of a specific node
     *
     * @param node The node ID to get neighbors for
     * @return const std::vector<NodeId>& Vector of neighbor node IDs
     */
    const std::vector<NodeId>& get_neighbors(const NodeId& node) const;

    /**
     * @brief Get read-only access to the adjacency map
     *
     * @return const AdjacencyMap& Read-only reference to the internal adjacency map
     */
    const AdjacencyMap& get_adjacency_map() const;

    /**
     * @brief Print adjacency map for debugging
     *
     * Prints the graph structure showing each node and its neighbors.
     * Useful for debugging mapping failures.
     *
     * @param graph_name Name to identify this graph in the output
     */
    void print_adjacency_map(const std::string& graph_name = "Graph", bool quiet_mode = false) const;

private:
    AdjacencyMap adj_map_;
    std::vector<NodeId> nodes_cache_;
};

std::map<MeshId, AdjacencyGraph<FabricNodeId>> build_adjacency_graph_logical(const MeshGraph& mesh_graph);
std::map<MeshId, AdjacencyGraph<FabricNodeId>> build_adjacency_graph_logical(const MeshGraphDescriptor& mgd);

std::map<MeshId, AdjacencyGraph<tt::tt_metal::AsicID>> build_adjacency_graph_physical(
    tt::tt_metal::ClusterType cluster_type,
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>>& asic_id_to_mesh_rank);

/**
 * @brief Unified constraint system for topology mapping
 *
 * MappingConstraints represents all constraints internally as trait maps. Both trait-based
 * constraints (one-to-many) and explicit pair constraints (one-to-one) are unified into a
 * single intersection-based representation.
 *
 * @tparam TargetNode The type of nodes in the target graph
 * @tparam GlobalNode The type of nodes in the global graph
 */
template <typename TargetNode, typename GlobalNode>
class MappingConstraints {
public:
    /// A set of (target, global) node pairs used in one cardinality constraint.
    using CardinalityPairSet = std::set<std::pair<TargetNode, GlobalNode>>;
    /// One cardinality constraint: (pair_set, min_count).
    using CardinalityConstraintEntry = std::pair<CardinalityPairSet, size_t>;
    /// The full list of cardinality constraints stored by this object.
    using CardinalityConstraintList = std::vector<CardinalityConstraintEntry>;

    /**
     * @brief Construct empty constraints
     */
    MappingConstraints() = default;

    /**
     * @brief Constructor from sets of constraint pairs
     *
     * Converts pairs into the internal mapping representation.
     *
     * @param required_constraints Set of required constraint pairs (target, global)
     * @param preferred_constraints Set of preferred constraint pairs (target, global)
     */
    MappingConstraints(
        const std::set<std::pair<TargetNode, GlobalNode>>& required_constraints,
        const std::set<std::pair<TargetNode, GlobalNode>>& preferred_constraints = {});

    /**
     * @brief Add required trait-based constraint (one-to-many)
     *
     * Constrains target nodes with trait value T to only map to global nodes with same trait value T.
     * All constraints are intersected - a target node must satisfy ALL required constraints simultaneously.
     *
     * @tparam TraitType The type of the trait value (must be explicitly specified)
     * @param target_traits Map from target nodes to their trait values
     * @param global_traits Map from global nodes to their trait values
     * @return true if constraint was successfully added, false if constraint causes empty valid mappings
     */
    template <typename TraitType>
    bool add_required_trait_constraint(
        const std::map<TargetNode, TraitType>& target_traits, const std::map<GlobalNode, TraitType>& global_traits);

    /**
     * @brief Add preferred trait-based constraint (one-to-many)
     *
     * Constrains target nodes with trait value T to prefer mapping to global nodes with same trait value T.
     * Preferred constraints guide the solver but don't restrict valid mappings.
     *
     * @tparam TraitType The type of the trait value (must be explicitly specified)
     * @param target_traits Map from target nodes to their trait values
     * @param global_traits Map from global nodes to their trait values
     */
    template <typename TraitType>
    void add_preferred_trait_constraint(
        const std::map<TargetNode, TraitType>& target_traits, const std::map<GlobalNode, TraitType>& global_traits);

    /**
     * @brief Add explicit required constraint (one-to-one)
     *
     * Pins a specific target node to a specific global node.
     * Intersects with existing constraints. Throws TT_THROW if constraint causes conflicts.
     *
     * @param target_node The target node to constrain
     * @param global_node The global node it must map to
     * @return true if constraint was successfully added, false if constraint causes empty valid mappings
     */
    bool add_required_constraint(TargetNode target_node, GlobalNode global_node);

    /**
     * @brief Add explicit required constraint (one-to-many for target node)
     *
     * Constrains a specific target node to map to any of the provided global nodes.
     * Intersects with existing constraints.
     *
     * @param target_node The target node to constrain
     * @param global_nodes The set of global nodes it can map to
     * @return true if constraint was successfully added, false if constraint causes empty valid mappings
     */
    bool add_required_constraint(TargetNode target_node, const std::set<GlobalNode>& global_nodes);

    /**
     * @brief Add explicit required constraint (one-to-many for global node)
     *
     * Constrains multiple target nodes to map to a specific global node.
     * Intersects with existing constraints.
     *
     * @param target_nodes The set of target nodes to constrain
     * @param global_node The global node they must map to
     * @return true if constraint was successfully added, false if constraint causes empty valid mappings
     */
    bool add_required_constraint(const std::set<TargetNode>& target_nodes, GlobalNode global_node);

    /**
     * @brief Add explicit required constraint (many-to-many)
     *
     * Constrains multiple target nodes to map to any of the provided global nodes.
     * This creates a many-to-many relationship: any target node from the set can map
     * to any global node from the set. Intersects with existing constraints for each target.
     *
     * @param target_nodes The set of target nodes to constrain
     * @param global_nodes The set of global nodes they can map to
     * @return true if constraint was successfully added, false if constraint causes empty valid mappings
     */
    bool add_required_constraint(const std::set<TargetNode>& target_nodes, const std::set<GlobalNode>& global_nodes);

    /**
     * @brief Add explicit preferred constraint (one-to-one)
     *
     * Suggests a mapping but doesn't restrict valid mappings.
     * The solver can still choose other nodes if needed.
     *
     * @param target_node The target node
     * @param global_node The preferred global node to map to
     */
    void add_preferred_constraint(TargetNode target_node, GlobalNode global_node);

    /**
     * @brief Add explicit preferred constraint (one-to-many for target node)
     *
     * Suggests that a specific target node prefers mapping to any of the provided global nodes.
     * Intersects with existing preferred constraints. Doesn't restrict valid mappings.
     *
     * @param target_node The target node
     * @param global_nodes The set of preferred global nodes it can map to
     */
    void add_preferred_constraint(TargetNode target_node, const std::set<GlobalNode>& global_nodes);

    /**
     * @brief Add explicit preferred constraint (one-to-many for global node)
     *
     * Suggests that multiple target nodes prefer mapping to a specific global node.
     * Intersects with existing preferred constraints. Doesn't restrict valid mappings.
     *
     * @param target_nodes The set of target nodes
     * @param global_node The preferred global node they should map to
     */
    void add_preferred_constraint(const std::set<TargetNode>& target_nodes, GlobalNode global_node);

    /**
     * @brief Add explicit forbidden constraint (one-to-one)
     *
     * Forbids a specific target node from mapping to a specific global node.
     * Removes the mapping from valid mappings.
     *
     * @param target_node The target node to constrain
     * @param global_node The global node it cannot map to
     * @return true if constraint was successfully added, false if constraint contradicts required constraint or causes
     * empty valid mappings
     */
    bool add_forbidden_constraint(TargetNode target_node, GlobalNode global_node);

    /**
     * @brief Add explicit forbidden constraint (one-to-many for target node)
     *
     * Forbids a specific target node from mapping to any of the provided global nodes.
     * Removes the mappings from valid mappings.
     *
     * @param target_node The target node to constrain
     * @param global_nodes The set of global nodes it cannot map to
     * @return true if constraint was successfully added, false if constraint contradicts required constraint or causes
     * empty valid mappings
     */
    bool add_forbidden_constraint(TargetNode target_node, const std::set<GlobalNode>& global_nodes);

    /**
     * @brief Add explicit forbidden constraint (one-to-many for global node)
     *
     * Forbids multiple target nodes from mapping to a specific global node.
     * Removes the mappings from valid mappings.
     *
     * @param target_nodes The set of target nodes to constrain
     * @param global_node The global node they cannot map to
     * @return true if constraint was successfully added, false if constraint contradicts required constraint or causes
     * empty valid mappings
     */
    bool add_forbidden_constraint(const std::set<TargetNode>& target_nodes, GlobalNode global_node);

    /**
     * @brief Add explicit forbidden constraint (two full lists, Cartesian product)
     *
     * Forbids every (target, global) pair in `target_nodes` × `global_nodes`. Both sets must be
     * non-empty explicit lists; empty target or global set is rejected (returns false, logged).
     *
     * @param target_nodes Non-empty set of target nodes
     * @param global_nodes Non-empty set of global nodes (e.g. chips / ASICs)
     * @return true if all pairs were added and constraints remain valid
     */
    bool add_forbidden_constraint(const std::set<TargetNode>& target_nodes, const std::set<GlobalNode>& global_nodes);

    /**
     * @brief Add cardinality constraint (at-least-N constraint)
     *
     * Requires that at least `min_count` of the provided (target, global) mapping pairs
     * must be satisfied in the final solution. This allows expressing constraints like
     * "at least 1 of these mappings must be satisfied" or "at least 2 of these mappings".
     *
     * Example: If you want nodes x, y to map to values 1, 2, 3, 4, but only need one
     * of them to map, you can add all possible pairs and set min_count=1:
     *   add_cardinality_constraint({{x,1}, {x,2}, {x,3}, {x,4}, {y,1}, {y,2}, {y,3}, {y,4}}, 1)
     *
     * @param mapping_pairs Set of (target_node, global_node) pairs that form the constraint group
     * @param min_count Minimum number of pairs that must be satisfied (default: 1)
     * @return true if constraint was successfully added, false if constraint is invalid or unsatisfiable
     */
    bool add_cardinality_constraint(const CardinalityPairSet& mapping_pairs, size_t min_count = 1);

    /**
     * @brief Add many-to-many cardinality constraint (convenience method)
     *
     * Generates all possible (target, global) pairs from the Cartesian product of the two sets
     * and requires that at least `min_count` of these pairs must be satisfied in the final solution.
     * This is a convenience method that avoids manually listing all pairs.
     *
     * Example: If you want nodes {x, y} to map to values {1, 2, 3, 4}, but only need 2
     * of the possible mappings to be satisfied:
     *   add_cardinality_constraint({x, y}, {1, 2, 3, 4}, 2)
     * This is equivalent to:
     *   add_cardinality_constraint({{x,1}, {x,2}, {x,3}, {x,4}, {y,1}, {y,2}, {y,3}, {y,4}}, 2)
     *
     * @param target_nodes Set of target nodes
     * @param global_nodes Set of global nodes
     * @param min_count Minimum number of pairs that must be satisfied (default: 1)
     * @return true if constraint was successfully added, false if constraint is invalid or unsatisfiable
     */
    bool add_cardinality_constraint(
        const std::set<TargetNode>& target_nodes, const std::set<GlobalNode>& global_nodes, size_t min_count = 1);

    /**
     * @brief Get valid mappings for a specific target node
     *
     * @param target The target node
     * @return const std::set<GlobalNode>& Set of global nodes this target can map to
     */
    const std::set<GlobalNode>& get_valid_mappings(TargetNode target) const;

    /**
     * @brief Get preferred mappings for a specific target node
     *
     * @param target The target node
     * @return const std::set<GlobalNode>& Set of preferred global nodes for this target
     */
    const std::set<GlobalNode>& get_preferred_mappings(TargetNode target) const;

    /**
     * @brief Check if a specific mapping is valid
     *
     * @param target The target node
     * @param global The global node
     * @return true if the mapping satisfies all required constraints, false otherwise
     */
    bool is_valid_mapping(TargetNode target, GlobalNode global) const;

    /**
     * @brief Get all valid mappings (for solver access)
     *
     * @return const std::map<TargetNode, std::set<GlobalNode>>& Map of all valid mappings
     */
    const std::map<TargetNode, std::set<GlobalNode>>& get_valid_mappings() const;

    /**
     * @brief Get all preferred mappings (for solver access)
     *
     * @return const std::map<TargetNode, std::set<GlobalNode>>& Map of all preferred mappings
     */
    const std::map<TargetNode, std::set<GlobalNode>>& get_preferred_mappings() const;

    /**
     * @brief Get all cardinality constraints (for solver access)
     *
     * @return Vector of (mapping_pairs, min_count) tuples representing cardinality constraints
     */
    const CardinalityConstraintList& get_cardinality_constraints() const;

    /**
     * @brief Set same-group constraint (for UNSET host rank binding)
     *
     * Ensures that targets in the same target group map to globals in one physical partition (one
     * set in global_groups). Each non-empty target group is matched to a distinct non-empty global
     * group; the solver chooses which group pairs with which (no index alignment). Apply required
     * constraints (e.g. rank must map to a specific host's ASICs) after this for explicit bindings.
     *
     * @param target_groups Vector of sets; each set is target nodes (e.g. fabric nodes per rank)
     * @param global_groups Vector of sets; each set is global nodes (e.g. ASICs per host)
     * @return false if this would contradict required, forbidden, or same-rank feasibility (state unchanged)
     */
    bool set_same_rank_groups_constraint(
        const std::vector<std::set<TargetNode>>& target_groups, const std::vector<std::set<GlobalNode>>& global_groups);

    const std::vector<std::set<TargetNode>>& get_same_rank_target_groups() const { return same_rank_target_groups_; }
    const std::vector<std::set<GlobalNode>>& get_same_rank_global_groups() const { return same_rank_global_groups_; }

    /**
     * @brief Opt-in objective: minimize the number of distinct same-rank GLOBAL groups (e.g. host partitions)
     * that the mapping touches.
     *
     * When enabled (and same-rank global groups are present), the SAT backend adds a host-usage budget: it tries
     * to confine the whole mapping to the provably-minimal number of groups (ceil(num_targets / max_group_size))
     * and walks the budget upward only if that is infeasible. This packs connected targets (e.g. a pipeline) onto
     * the fewest hosts. It is a best-effort objective: if no budget is satisfiable the solver falls back to an
     * unconstrained solve, so enabling it can never turn a solvable instance UNSAT. The DFS backend approximates
     * the same goal via a host-affinity value-ordering bias. Off by default; intended for inter-mesh mapping.
     */
    void set_minimize_same_rank_groups_used(bool enable) { minimize_same_rank_groups_used_ = enable; }
    bool minimize_same_rank_groups_used() const { return minimize_same_rank_groups_used_; }

    /**
     * @brief HARD cap on the number of distinct same-rank global groups (host partitions) the mapping may occupy.
     *
     * When > 0 (and same-rank global groups are present), the SAT backend adds a HARD "at most k groups occupied"
     * cardinality constraint over per-group occupancy indicators: the mapping MUST fit inside k hosts, but the
     * solver is free to choose WHICH k (any combination) -- unlike pinning to a specific cover. Set k to the
     * capacity lower bound (ceil(num_targets / max_group_size)) to force the minimum host count. If the cap is
     * unsatisfiable, the solver backs down to the soft minimize objective (set_minimize_same_rank_groups_used)
     * when that is also enabled, so requesting both gives "hard-if-possible, else minimize best-effort". 0 = no cap.
     */
    void set_max_same_rank_groups_used(std::size_t k) { max_same_rank_groups_used_ = k; }
    std::size_t max_same_rank_groups_used() const { return max_same_rank_groups_used_; }

    /**
     * @brief Get forbidden (target, global) pairs that are invalid even when no required constraints exist
     *
     * Used when add_forbidden_constraint is called for a target with no valid_mappings_ entry.
     * These pairs are stored separately and exclude specific mappings without requiring seeding.
     *
     * @return const std::set<std::pair<TargetNode, GlobalNode>>& Set of forbidden pairs
     */
    const std::set<std::pair<TargetNode, GlobalNode>>& get_forbidden_pairs() const;

    /**
     * @brief Validate constraints - returns false if invalid
     *
     * If saved_state is provided and validation fails, restores the saved state before returning false.
     * Per-target: std::nullopt means the target had no valid_mappings_ entry before the attempted change
     * (restore erases the key); a set value restores that target's previous allowed globals.
     *
     * @param saved_state Optional pointer to saved state to restore on failure
     * @return true if constraints are valid, false otherwise
     */
    bool validate(const std::map<TargetNode, std::optional<std::set<GlobalNode>>>* saved_state = nullptr);

    /**
     * @brief Set quiet mode for constraint validation messages
     *
     * When quiet mode is enabled, overconstrained validation messages are logged at debug level
     * instead of info level to reduce verbosity.
     *
     * @param quiet_mode If true, suppress info-level constraint validation messages
     */
    void set_quiet_mode(bool quiet_mode) const;

    /**
     * @brief Print mapping constraint maps for debugging
     *
     * Prints valid (required), preferred, forbidden, cardinality, and same-rank constraints in a
     * list form analogous to printing an adjacency map (per-target lines with neighbor-like sets).
     *
     * @param label Section title prefix (e.g. "Mesh 0 constraints")
     * @param quiet_mode If true, detailed lines are logged at debug level instead of info
     */
    void print_mapping_constraint_maps(const std::string& label = "Mapping constraints", bool quiet_mode = false) const;

private:
    // Internal representation: intersection of all constraints
    std::map<TargetNode, std::set<GlobalNode>> valid_mappings_;      // Required constraints
    std::map<TargetNode, std::set<GlobalNode>> preferred_mappings_;  // Preferred constraints

    // Forbidden (target, global) pairs - used when target has no valid_mappings_ entry.
    // Allows add_forbidden_constraint to work without seeding valid_mappings_.
    std::set<std::pair<TargetNode, GlobalNode>> forbidden_pairs_;

    // Cardinality constraints: each entry requires that at least min_count of its
    // (target, global) node pairs must be satisfied by the mapping.
    CardinalityConstraintList cardinality_constraints_;

    // Same-group constraint: targets in a target group map to at most one global group
    std::vector<std::set<TargetNode>> same_rank_target_groups_;
    std::vector<std::set<GlobalNode>> same_rank_global_groups_;

    // Opt-in objective: minimize number of distinct same-rank global groups (host partitions) used.
    bool minimize_same_rank_groups_used_ = false;

    // Opt-in HARD cap: at most this many distinct same-rank global groups may be occupied (0 = no cap).
    std::size_t max_same_rank_groups_used_ = 0;

    // Track which global nodes are exclusively reserved by many-to-many constraints
    // Maps global node -> set of target nodes that are allowed to map to it via many-to-many constraints
    std::map<GlobalNode, std::set<TargetNode>> reserved_global_nodes_;

    // Quiet mode flag - mutable so it can be set even on const objects
    mutable bool quiet_mode_ = false;

    // Helper to intersect two sets
    static std::set<GlobalNode> intersect_sets(const std::set<GlobalNode>& set1, const std::set<GlobalNode>& set2);

    // Validate that all cardinality constraints are compatible with required constraints
    // and that they are satisfiable together
    bool validate_cardinality_constraints() const;

    // Same-rank feasibility: every non-empty logical target group must have at least one physical
    // host partition where all members still allow a mapping (forbidden + valid_mappings / staged
    // rules). Multiple target groups may share the same partition (e.g. several mesh_host_ranks
    // carved from one galaxy host); partitions are not required to be distinct.
    bool validate_same_rank_groups_feasible() const;
};

/**
 * @brief Mode for connection count validation
 */
enum class ConnectionValidationMode {
    /// Strict mode: require exact channel counts, fail if not met
    STRICT,
    /// Relaxed mode: allow insufficient channels (warnings) but prefer mappings with better-matched physical link
    /// capacity. Current DFS biases search via candidate ordering; SAT/MaxSAT backend should add automatic weighted
    /// soft objectives for channel alignment (see migration plan).
    RELAXED
};

/**
 * @brief Search backend for solve_topology_mapping
 *
 * Use Dfs or Sat for explicit control (e.g. unit tests). Auto uses a size-based heuristic: small problems
 * (n_target * n_global < threshold) use DFS for minimal overhead, while large problems use SAT for
 * superior search efficiency. The environment variable TT_TOPOLOGY_SOLVER_ENGINE can override Auto:
 * set to "sat" to force SAT everywhere, or "dfs" to force DFS everywhere.
 */
enum class TopologyMappingSolverEngine {
    Auto,
    Dfs,
    Sat,
};

/**
 * @brief Result of topology mapping operation
 *
 * Contains the mapping result, success status, error messages, and statistics
 * about the constraint satisfaction process.
 *
 * @tparam TargetNode The type used to identify nodes in the target graph
 * @tparam GlobalNode The type used to identify nodes in the global graph
 */
template <typename TargetNode, typename GlobalNode>
struct MappingResult {
    /// Whether the mapping was successful
    bool success = false;

    /// Error message if mapping failed
    std::string error_message;

    /// Warning messages (e.g., relaxed mode connection count mismatches)
    std::vector<std::string> warnings;

    /// Mapping from target nodes to global nodes
    std::map<TargetNode, GlobalNode> target_to_global;

    /// Reverse mapping from global nodes to target nodes
    std::map<GlobalNode, TargetNode> global_to_target;

    /// Statistics about constraint satisfaction
    struct ConstraintStats {
        size_t required_satisfied = 0;   ///< Number of required constraints satisfied
        size_t preferred_satisfied = 0;  ///< Number of preferred constraints satisfied
        size_t preferred_total = 0;      ///< Total number of preferred constraints
    } constraint_stats;

    /// Statistics about the solving process
    struct Stats {
        size_t dfs_calls = 0;                      ///< Number of DFS calls made
        size_t backtrack_count = 0;                ///< Number of backtracks performed
        size_t memoization_hits = 0;               ///< Number of times memoization cache was hit
        std::chrono::microseconds elapsed_time{};  ///< Time taken to solve (microsecond resolution)
    } stats;
};

/**
 * @brief Print mapping result for debugging
 *
 * Prints the mapping showing which target nodes map to which global nodes,
 * along with warnings, statistics, and other diagnostic information.
 *
 * Template parameters are automatically deduced from the function arguments,
 * so explicit template parameters are not required when calling this function.
 *
 * @tparam TargetNode The type used to identify nodes in the target graph (deduced)
 * @tparam GlobalNode The type used to identify nodes in the global graph (deduced)
 * @param result The mapping result to print
 *
 * @example
 * ```cpp
 * MappingResult<FabricNodeId, AsicID> result = solve_topology_mapping(...);
 * print_mapping_result(result);  // No template parameters needed
 * ```
 */
template <typename TargetNode, typename GlobalNode>
void print_mapping_result(const MappingResult<TargetNode, GlobalNode>& result);

/**
 * @brief Solve topology mapping using constraint satisfaction
 *
 * Stateless function that performs constraint satisfaction search to find a valid
 * mapping from target graph to global graph. Enforces required constraints first,
 * then optimizes for preferred constraints. In RELAXED mode, the search also favors
 * embeddings that better match target edge channel counts on the physical graph
 * (more capacity satisfied is preferred over less), without requiring explicit
 * preferred constraints for that behavior.
 *
 * @tparam TargetNode The type used to identify nodes in the target graph (must be explicitly specified)
 * @tparam GlobalNode The type used to identify nodes in the global graph (must be explicitly specified)
 * @param target_graph The target graph (subgraph pattern to find)
 * @param global_graph The global graph (larger host graph that contains the target)
 * @param constraints The mapping constraints to satisfy
 * @param connection_validation_mode STRICT fails on insufficient channels; RELAXED allows them but still prefers
 *        stronger channel alignment among feasible mappings (default: RELAXED)
 * @param quiet_mode If true, log errors at debug level instead of error level (useful for auto-discovery)
 * @param solver_engine Auto uses TT_TOPOLOGY_SOLVER_ENGINE; Dfs/Sat force that backend regardless of env.
 * @return MappingResult containing success status, bidirectional mappings, and warnings
 */
template <typename TargetNode, typename GlobalNode>
MappingResult<TargetNode, GlobalNode> solve_topology_mapping(
    const AdjacencyGraph<TargetNode>& target_graph,
    const AdjacencyGraph<GlobalNode>& global_graph,
    const MappingConstraints<TargetNode, GlobalNode>& constraints,
    ConnectionValidationMode connection_validation_mode = ConnectionValidationMode::RELAXED,
    bool quiet_mode = false,
    TopologyMappingSolverEngine solver_engine = TopologyMappingSolverEngine::Auto);

/**
 * @brief Find up to N distinct valid topology mappings.
 *
 * Runs the solver repeatedly (using blocking clauses for SAT, continued backtracking for DFS)
 * collecting distinct solutions until either max_solutions mappings have been found or the
 * problem space is exhausted. Each returned MappingResult is individually validated.
 *
 * When TopologyMappingSolverEngine::Sat (or Auto when it selects SAT) is in use, every enumeration — including
 * max_solutions > 1 and solve_topology_mapping_all — uses CaDiCaL incrementally: hard constraints are encoded once,
 * then blocking clauses are appended between solves (see topology_sat_search_n). DFS is used only when the engine
 * resolves to DFS.
 *
 * @param target_graph The target (sub-)graph pattern to embed
 * @param global_graph The host graph to embed into
 * @param constraints Mapping constraints
 * @param max_solutions Maximum number of solutions to return (0 means enumerate up to the
 *        implementation-defined safety limit; values above that limit are clamped the same way)
 * @param connection_validation_mode STRICT or RELAXED channel validation
 * @param quiet_mode Suppress verbose logging
 * @param solver_engine Which backend to use
 * @param unique_shapes If true, count solutions by the set of global nodes used (order-independent); permutations on
 *        the same global set share one slot. For SAT enumeration this is enforced with extra CNF clauses so the
 *        solver skips entire automorphism classes per model. For DFS, equivalent pruning is applied where possible.
 * @return Vector of up to max_solutions valid MappingResults (may be empty if no solution exists)
 */
template <typename TargetNode, typename GlobalNode>
std::vector<MappingResult<TargetNode, GlobalNode>> solve_topology_mapping_n(
    const AdjacencyGraph<TargetNode>& target_graph,
    const AdjacencyGraph<GlobalNode>& global_graph,
    const MappingConstraints<TargetNode, GlobalNode>& constraints,
    size_t max_solutions,
    ConnectionValidationMode connection_validation_mode = ConnectionValidationMode::RELAXED,
    bool quiet_mode = false,
    TopologyMappingSolverEngine solver_engine = TopologyMappingSolverEngine::Auto,
    bool unique_shapes = false);

/**
 * @brief Find all distinct valid topology mappings up to the implementation enumeration limit.
 *
 * Equivalent to solve_topology_mapping_n(..., 0, ...) (see max_solutions semantics there).
 *
 * @param target_graph The target (sub-)graph pattern to embed
 * @param global_graph The host graph to embed into
 * @param constraints Mapping constraints
 * @param connection_validation_mode STRICT or RELAXED channel validation
 * @param quiet_mode Suppress verbose logging
 * @param solver_engine Which backend to use
 * @param unique_shapes See solve_topology_mapping_n
 * @return Vector of all valid MappingResults found within that limit. If the result count equals the
 *         implementation enumeration cap, a warning is logged: more solutions may exist.
 */
template <typename TargetNode, typename GlobalNode>
std::vector<MappingResult<TargetNode, GlobalNode>> solve_topology_mapping_all(
    const AdjacencyGraph<TargetNode>& target_graph,
    const AdjacencyGraph<GlobalNode>& global_graph,
    const MappingConstraints<TargetNode, GlobalNode>& constraints,
    ConnectionValidationMode connection_validation_mode = ConnectionValidationMode::RELAXED,
    bool quiet_mode = false,
    TopologyMappingSolverEngine solver_engine = TopologyMappingSolverEngine::Auto,
    bool unique_shapes = false);

namespace detail {
inline std::vector<int> topology_mapping_shape_key(const std::vector<int>& mapping) {
    std::vector<int> key;
    key.reserve(mapping.size());
    for (int g : mapping) {
        if (g >= 0) {
            key.push_back(g);
        }
    }
    std::sort(key.begin(), key.end());
    return key;
}

bool topology_mapping_should_use_sat_engine(
    TopologyMappingSolverEngine engine, size_t n_target = 0, size_t n_global = 0);

/** @see TT_TOPOLOGY_SOLVER_ENGINE in solve_topology_mapping documentation. */
inline bool topology_mapping_use_sat_engine();

/**
 * @brief Indexed graph representation for efficient lookups
 *
 * Converts AdjacencyGraph into index-based representation for O(1) lookups.
 * Stores deduplicated, sorted adjacency lists and connection counts.
 */
template <typename TargetNode, typename GlobalNode>
struct GraphIndexData {
    // Node vectors
    std::vector<TargetNode> target_nodes;
    std::vector<GlobalNode> global_nodes;

    // Index mappings
    std::map<TargetNode, size_t> target_to_idx;
    std::map<GlobalNode, size_t> global_to_idx;

    // Adjacency index vectors (deduplicated, sorted)
    std::vector<std::vector<size_t>> target_adj_idx;
    std::vector<std::vector<size_t>> global_adj_idx;

    // Connection count maps (for strict mode / multi-edge support)
    std::vector<std::map<size_t, size_t>> target_conn_count;
    std::vector<std::map<size_t, size_t>> global_conn_count;

    // Degree vectors
    std::vector<size_t> target_deg;
    std::vector<size_t> global_deg;

    size_t n_target = 0;
    size_t n_global = 0;

    /**
     * @brief Construct GraphIndexData from AdjacencyGraph inputs
     *
     * Builds indexed representation from target and global graphs.
     */
    GraphIndexData(const AdjacencyGraph<TargetNode>& target_graph, const AdjacencyGraph<GlobalNode>& global_graph);

    /**
     * @brief Print node degrees for debugging
     *
     * Prints the degree of each node in both target and global graphs.
     * Useful for understanding graph structure during mapping.
     */
    void print_node_degrees() const;

    /**
     * @brief Print adjacency maps for debugging
     *
     * Prints the adjacency structure of both target and global graphs.
     * Useful for debugging mapping failures.
     */
    void print_adjacency_maps() const;
};

/// A cardinality constraint in index form: at least @c min_count of the
/// (target_idx, global_idx) @c pairs must be satisfied by the final mapping.
struct IndexedCardinalityConstraint {
    std::set<std::pair<size_t, size_t>> pairs;  ///< (target_idx, global_idx) index pairs
    size_t min_count = 0;                        ///< Minimum number of pairs that must be mapped
};

/**
 * @brief Indexed constraint representation for efficient lookups
 *
 * Converts MappingConstraints into index-based representation for O(1) constraint checks.
 * Stores restricted and preferred mappings as index vectors.
 */
template <typename TargetNode, typename GlobalNode>
struct ConstraintIndexData {
    // Restricted mappings: target_idx -> vector of valid global_indices
    // If empty for a target_idx, that target can map to any global node
    std::vector<std::vector<size_t>> restricted_global_indices;

    // Forbidden mappings: target_idx -> sorted vector of forbidden global_indices
    // Pairs from add_forbidden_constraint when target had no valid_mappings_ entry
    std::vector<std::vector<size_t>> forbidden_global_indices;

    // Preferred mappings: target_idx -> vector of preferred global_indices
    // Used for optimization, doesn't restrict valid mappings
    std::vector<std::vector<size_t>> preferred_global_indices;

    // Cardinality constraints: each entry requires that at least min_count of its
    // (target_idx, global_idx) pairs are satisfied by the mapping.
    std::vector<IndexedCardinalityConstraint> cardinality_constraints;

    // Same-group: target_idx/global_idx -> group_id (-1 or SIZE_MAX if not in any group)
    std::vector<int> global_to_same_rank_group;
    std::vector<std::set<size_t>> same_rank_groups;
    std::vector<size_t> target_to_group;

    // Opt-in objective: minimize the number of distinct same-rank global groups (host partitions) used.
    bool minimize_same_rank_groups_used = false;

    // Opt-in HARD cap: at most this many distinct same-rank global groups may be occupied (0 = no cap).
    std::size_t max_same_rank_groups_used = 0;

    /**
     * @brief Construct ConstraintIndexData from MappingConstraints and GraphIndexData
     *
     * Builds indexed constraint representation from constraints and graph data.
     */
    ConstraintIndexData(
        const MappingConstraints<TargetNode, GlobalNode>& constraints,
        const GraphIndexData<TargetNode, GlobalNode>& graph_data);

    /**
     * @brief Check if cardinality constraints are satisfied with current mapping
     *
     * @param mapping Complete mapping
     * @return true if all cardinality constraints are satisfied
     */
    bool check_cardinality_constraints(const std::vector<int>& mapping) const;

    /**
     * @brief Check if cardinality constraints can still be satisfied with current mapping
     *
     * @param mapping Current partial mapping
     * @return true if constraints can still be satisfied
     */
    bool can_satisfy_cardinality_constraints(const std::vector<int>& mapping) const;

    /**
     * @brief Get single required mapping for a target node (for pinning/pre-assignment)
     *
     * @param target_idx Target node index
     * @return Global node index if exactly one required constraint exists, SIZE_MAX otherwise
     */
    size_t get_single_required_mapping(size_t target_idx) const;

    /**
     * @brief Compute constraint statistics from a mapping
     *
     * @param mapping Complete mapping
     * @param graph_data Graph index data
     * @return Tuple of (required_satisfied, preferred_satisfied, preferred_total)
     */
    std::tuple<size_t, size_t, size_t> compute_constraint_stats(
        const std::vector<int>& mapping, const GraphIndexData<TargetNode, GlobalNode>& graph_data) const;

    // Helper: check if mapping is valid
    bool is_valid_mapping(size_t target_idx, size_t global_idx) const;

    /**
     * @brief Check if assigning (target_idx, global_idx) satisfies same-rank groups constraint
     *
     * Ensures no target-group boundary splitting: all other targets in target_idx's group
     * that are already assigned must map to globals in the same global group as global_idx.
     * Multiple target groups may share a global group.
     */
    bool check_same_rank_constraint(
        size_t target_idx, size_t global_idx, const std::vector<int>& mapping, const std::vector<bool>& used) const;

    // Helper: get candidates for target node
    // Returns restricted candidates if available, otherwise returns empty vector (meaning all are valid)
    const std::vector<size_t>& get_candidates(size_t target_idx) const;

    /**
     * @brief Print resolved constraint maps (indices resolved to node IDs) for debugging
     *
     * Prints the indexed restricted, forbidden, preferred, cardinality, and same-rank state using
     * graph_data to resolve indices to TargetNode / GlobalNode, similar in spirit to
     * GraphIndexData::print_adjacency_maps().
     *
     * @param graph_data Graph index data used when this ConstraintIndexData was built
     * @param label Section title prefix
     * @param quiet_mode If true, detailed lines are logged at debug level instead of info
     */
    void print_resolved_mapping_constraint_maps(
        const GraphIndexData<TargetNode, GlobalNode>& graph_data,
        const std::string& label = "Resolved mapping constraints",
        bool quiet_mode = false) const;
};

/** SAT encoder state (no CaDiCaL types in the public header). */
struct TopologySatHardEncoding {
    bool trivial_unsat = false;
    std::string trivial_reason;
    std::vector<std::vector<size_t>> allowed_global_idx;
    std::vector<std::vector<int>> assign_lit;
};

/**
 * Index-only view of GraphIndexData for the SAT backend (implemented in topology_solver_sat.cpp).
 */
struct TopologySatGraphView {
    size_t n_target = 0;
    size_t n_global = 0;
    const std::vector<std::vector<size_t>>& target_adj_idx;
    const std::vector<std::vector<size_t>>& global_adj_idx;
    const std::vector<std::map<size_t, size_t>>& target_conn_count;
    const std::vector<std::map<size_t, size_t>>& global_conn_count;
    const std::vector<size_t>& target_deg;
    const std::vector<size_t>& global_deg;

    template <typename TargetNode, typename GlobalNode>
    explicit TopologySatGraphView(const GraphIndexData<TargetNode, GlobalNode>& g) :
        n_target(g.n_target),
        n_global(g.n_global),
        target_adj_idx(g.target_adj_idx),
        global_adj_idx(g.global_adj_idx),
        target_conn_count(g.target_conn_count),
        global_conn_count(g.global_conn_count),
        target_deg(g.target_deg),
        global_deg(g.global_deg) {}
};

struct TopologySatConstraintView {
    const std::vector<std::vector<size_t>>& restricted_global_indices;
    const std::vector<std::vector<size_t>>& forbidden_global_indices;
    const std::vector<std::vector<size_t>>& preferred_global_indices;
    const std::vector<IndexedCardinalityConstraint>& cardinality_constraints;
    const std::vector<int>& global_to_same_rank_group;
    const std::vector<std::set<size_t>>& same_rank_groups;
    const std::vector<size_t>& target_to_group;
    bool minimize_same_rank_groups_used = false;
    std::size_t max_same_rank_groups_used = 0;

    template <typename TargetNode, typename GlobalNode>
    explicit TopologySatConstraintView(const ConstraintIndexData<TargetNode, GlobalNode>& c) :
        restricted_global_indices(c.restricted_global_indices),
        forbidden_global_indices(c.forbidden_global_indices),
        preferred_global_indices(c.preferred_global_indices),
        cardinality_constraints(c.cardinality_constraints),
        global_to_same_rank_group(c.global_to_same_rank_group),
        same_rank_groups(c.same_rank_groups),
        target_to_group(c.target_to_group),
        minimize_same_rank_groups_used(c.minimize_same_rank_groups_used),
        max_same_rank_groups_used(c.max_same_rank_groups_used) {}

    bool is_valid_mapping(size_t target_idx, size_t global_idx) const {
        if (target_idx < forbidden_global_indices.size() && !forbidden_global_indices[target_idx].empty()) {
            const auto& forbidden = forbidden_global_indices[target_idx];
            if (std::binary_search(forbidden.begin(), forbidden.end(), global_idx)) {
                return false;
            }
        }
        if (target_idx >= restricted_global_indices.size() || restricted_global_indices[target_idx].empty()) {
            return true;
        }
        const auto& candidates = restricted_global_indices[target_idx];
        return std::binary_search(candidates.begin(), candidates.end(), global_idx);
    }
};

// Opaque SAT solver session — full definition is in the private
// topology_solver_sat_session.hpp to keep CaDiCaL out of the public API.
struct TopologySatSession;

void topology_sat_session_destroy(TopologySatSession* p) noexcept;

struct TopologySatSessionDeleter {
    void operator()(TopologySatSession* p) const noexcept { topology_sat_session_destroy(p); }
};

// Creates a new SAT session and encodes hard constraints into it.
// On success, enc is populated and a non-null session is returned.
// Returns nullptr if the constraint set is hard-infeasible (no encoding possible).
std::unique_ptr<TopologySatSession, TopologySatSessionDeleter> topology_sat_session_create_and_encode(
    const TopologySatGraphView& graph_data,
    const TopologySatConstraintView& constraint_data,
    TopologySatHardEncoding& enc,
    ConnectionValidationMode validation_mode = ConnectionValidationMode::RELAXED);

// Appends a blocking clause for raw_mapping to session. Returns false on failure.
bool topology_sat_session_add_blocking_clause(
    TopologySatSession* session, TopologySatHardEncoding& enc,
    const std::vector<int>& raw_mapping, bool unique_shapes);

// Runs one solve call and decodes the solution into raw_out.
// Returns false if UNSAT or decoding fails.
bool topology_sat_session_solve_and_decode(
    TopologySatSession* session, const TopologySatHardEncoding& enc, std::vector<int>& raw_out);

struct TopologySearchState;

bool topology_sat_search(
    const TopologySatGraphView& graph_data,
    const TopologySatConstraintView& constraint_data,
    ConnectionValidationMode validation_mode,
    bool quiet_mode,
    TopologySearchState& state);

bool topology_sat_search_n(
    const TopologySatGraphView& graph_data,
    const TopologySatConstraintView& constraint_data,
    ConnectionValidationMode validation_mode,
    size_t max_solutions,
    std::vector<std::vector<int>>& all_mappings_out,
    bool quiet_mode,
    bool unique_shapes,
    const std::vector<std::vector<int>>& initial_forbidden_shape_keys,
    TopologySearchState& state);


/**
 * @brief Unified heuristic for node selection and candidate generation
 *
 * Combines node selection and candidate generation with explicit priority:
 * 1. Hard constraints (must satisfy)
 * 2. Soft constraints (optimize for)
 * 3. Runtime optimization (minimize search tree)
 *
 * Non-templated class - types are deduced from GraphIndexData at method call time.
 */
class SearchHeuristic {
public:
    /**
     * @brief Result of node selection and candidate generation
     */
    struct SelectionResult {
        size_t target_idx = SIZE_MAX;    // Selected target node index (SIZE_MAX if none)
        std::vector<size_t> candidates;  // Valid candidates (ordered by cost, lower = better)
    };

    /**
     * @brief Select next target node and generate ordered candidates
     *
     * Uses integer cost scoring (lower cost = higher priority):
     * - Node cost combines: hard constraints → soft constraints → runtime optimization
     * - Candidate cost: filters by hard constraints, orders by soft + runtime
     *
     * Uses ConstraintIndexData for fast index-based lookups (no node type conversions needed).
     *
     * @param graph_data Graph index data
     * @param constraint_data Constraint index data (for fast lookups)
     * @param mapping Current partial mapping (mapping[i] = global_idx or -1)
     * @param used Which global nodes are already used (used[i] = true if assigned)
     * @param validation_mode Connection validation mode
     * @return SelectionResult with selected node and ordered candidates
     */
    template <typename TargetNode, typename GlobalNode>
    static SelectionResult select_and_generate_candidates(
        const GraphIndexData<TargetNode, GlobalNode>& graph_data,
        const ConstraintIndexData<TargetNode, GlobalNode>& constraint_data,
        const std::vector<int>& mapping,
        const std::vector<bool>& used,
        ConnectionValidationMode validation_mode);

    /**
     * @brief Check if candidate satisfies all hard constraints
     *
     * @return true if candidate should be included, false if should be filtered out
     *
     * Public so ConsistencyChecker can use it for forward consistency checking.
     */
    template <typename TargetNode, typename GlobalNode>
    static bool check_hard_constraints(
        size_t target_idx,
        size_t global_idx,
        const GraphIndexData<TargetNode, GlobalNode>& graph_data,
        const ConstraintIndexData<TargetNode, GlobalNode>& constraint_data,
        const std::vector<int>& mapping,
        ConnectionValidationMode validation_mode);

    /**
     * @brief Generate and order candidates for a target node
     *
     * Filters by hard constraints first, then orders by cost (lower = better)
     */
    template <typename TargetNode, typename GlobalNode>
    static std::vector<size_t> generate_ordered_candidates(
        size_t target_idx,
        const GraphIndexData<TargetNode, GlobalNode>& graph_data,
        const ConstraintIndexData<TargetNode, GlobalNode>& constraint_data,
        const std::vector<int>& mapping,
        const std::vector<bool>& used,
        ConnectionValidationMode validation_mode);

private:
    /**
     * @brief Compute cost for selecting a target node (lower = better)
     *
     * cost = (candidate_count * HARD_WEIGHT)
     *      - (preferred_count * SOFT_WEIGHT)
     *      - (mapped_neighbors * RUNTIME_WEIGHT)
     */
    template <typename TargetNode, typename GlobalNode>
    static int compute_node_cost(
        size_t target_idx,
        const GraphIndexData<TargetNode, GlobalNode>& graph_data,
        const ConstraintIndexData<TargetNode, GlobalNode>& constraint_data,
        const std::vector<int>& mapping,
        const std::vector<bool>& used,
        ConnectionValidationMode validation_mode);

    /**
     * @brief Compute cost for a valid candidate (lower = better)
     *
     * Only called for candidates that passed hard constraint checks.
     * cost = -is_preferred * SOFT_WEIGHT
     *      - channel_match_count * SOFT_WEIGHT
     *      + degree_gap * RUNTIME_WEIGHT
     */
    template <typename TargetNode, typename GlobalNode>
    static int compute_candidate_cost(
        size_t target_idx,
        size_t global_idx,
        const GraphIndexData<TargetNode, GlobalNode>& graph_data,
        const ConstraintIndexData<TargetNode, GlobalNode>& constraint_data,
        const std::vector<int>& mapping,
        ConnectionValidationMode validation_mode);

    // Cost weights (ensure hard >> host-affinity >> soft >> runtime)
    static constexpr int HARD_WEIGHT = 1000000;
    // Host-affinity (packing) bias: must dominate the softer channel/preferred biases so connected targets
    // consolidate onto the fewest host partitions, but stay well below HARD_WEIGHT so it never competes with
    // hard feasibility. Applied per same-host already-mapped neighbor.
    static constexpr int HOST_AFFINITY_WEIGHT = 10000;
    static constexpr int SOFT_WEIGHT = 1000;
    static constexpr int RUNTIME_WEIGHT = 1;
};

/**
 * @brief ConsistencyChecker validates partial mappings during DFS to prune invalid branches early
 *
 * Non-templated struct with templated methods - template types are deduced from GraphIndexData arguments.
 * This allows usage without explicit template parameters: ConsistencyChecker::check_local_consistency(...)
 */
struct ConsistencyChecker {
    /**
     * @brief Check local consistency: verify assignment is consistent with already-assigned neighbors
     *
     * Checks that if target node A is mapped to global X, and target node B (neighbor of A)
     * is mapped to global Y, then X and Y must be connected.
     * In STRICT mode: also checks channel counts are sufficient.
     *
     * @param target_idx Index of target node being assigned
     * @param global_idx Index of global node being assigned to
     * @param graph_data Indexed graph data
     * @param mapping Current partial mapping (target_idx -> global_idx)
     * @param validation_mode Channel validation mode
     * @return true if assignment is locally consistent
     */
    template <typename TargetNode, typename GlobalNode>
    static bool check_local_consistency(
        size_t target_idx,
        size_t global_idx,
        const GraphIndexData<TargetNode, GlobalNode>& graph_data,
        const std::vector<int>& mapping,
        ConnectionValidationMode validation_mode);

    /**
     * @brief Check forward consistency: ensure assignment leaves viable options for future neighbors
     *
     * Verifies that each unassigned neighbor of the target node has at least one viable candidate
     * among the unused neighbors of the global node.
     *
     * @param target_idx Index of target node being assigned
     * @param global_idx Index of global node being assigned to
     * @param graph_data Indexed graph data
     * @param constraint_data Indexed constraint data
     * @param mapping Current partial mapping
     * @param used Which global nodes are already used
     * @param validation_mode Channel validation mode
     * @return true if assignment leaves viable options for future neighbors
     */
    template <typename TargetNode, typename GlobalNode>
    static bool check_forward_consistency(
        size_t target_idx,
        size_t global_idx,
        const GraphIndexData<TargetNode, GlobalNode>& graph_data,
        const ConstraintIndexData<TargetNode, GlobalNode>& constraint_data,
        const std::vector<int>& mapping,
        const std::vector<bool>& used,
        ConnectionValidationMode validation_mode);

    /**
     * @brief Count unused global nodes reachable from a starting point
     *
     * Used for path graph fast path optimization to verify there are enough
     * unused nodes for remaining target nodes.
     *
     * @param start_global_idx Starting global node index
     * @param graph_data Indexed graph data (only uses global graph)
     * @param used Which global nodes are already used
     * @return Number of unused nodes reachable from start_global_idx
     */
    template <typename TargetNode, typename GlobalNode>
    static size_t count_reachable_unused(
        size_t start_global_idx,
        const GraphIndexData<TargetNode, GlobalNode>& graph_data,
        const std::vector<bool>& used);
};

template <typename TargetNode, typename GlobalNode>
struct PathGraphDetector;

/**
 * @brief Shared search state for DFS and SAT topology engines
 *
 * The `mapping` vector contains the best or final assignment (global index per target, or -1).
 * DFS fills partial progress on failure; SAT typically leaves -1 on failure.
 */
struct TopologySearchState {
    std::vector<int> mapping;                    // mapping[target_idx] = global_idx or -1
    std::vector<bool> used;                      // used[global_idx] = true if assigned
    std::unordered_set<uint64_t> failed_states;  // DFS memoization cache (unused by SAT)
    size_t dfs_calls = 0;                        // DFS call count (0 for SAT)
    size_t backtrack_count = 0;                  // DFS backtracks (0 for SAT)
    size_t memoization_hits = 0;                 // DFS memoization hits (0 for SAT)
    std::string error_message;                   // Error message if search fails
};

/**
 * @brief DFS search engine for topology mapping
 *
 * Implements backtracking search with memoization and consistency checking.
 * Uses SearchHeuristic for node selection and candidate generation.
 *
 * **Important**: Even if the search fails to find a complete valid mapping, the
 * state's `mapping` will contain the best/closest partial mapping found.
 * This allows users to see what progress was made and diagnose why the search failed.
 * The MappingValidator will save this partial mapping in the result even if validation fails.
 */
template <typename TargetNode, typename GlobalNode>
class DFSSearchEngine {
public:
    using SearchState = TopologySearchState;

    /**
     * @brief Start DFS search
     *
     * **Note**: Even if this returns false (search failed), the internal state's mapping will contain
     * the best partial mapping found, which will be saved by MappingValidator for debugging.
     *
     * @param graph_data Indexed graph data
     * @param constraint_data Indexed constraint data (includes all constraint information)
     * @param validation_mode Connection validation mode
     * @return true if complete valid mapping found, false otherwise (but state still has best found)
     */
    bool search(
        const GraphIndexData<TargetNode, GlobalNode>& graph_data,
        const ConstraintIndexData<TargetNode, GlobalNode>& constraint_data,
        ConnectionValidationMode validation_mode,
        bool quiet_mode = false);

    /**
     * @brief Search for up to max_solutions distinct complete mappings using DFS with backtracking.
     *
     * Unlike search(), this method does NOT stop at the first solution. At each base-case
     * (all targets assigned) the mapping is pushed to all_mappings_out and the DFS continues
     * backtracking to look for additional solutions. Memoization of failed states is disabled
     * because a state that reaches one solution is not "failed" and should not prune other paths.
     * Stops early once all_mappings_out.size() >= max_solutions.
     *
     * @param graph_data Indexed graph data
     * @param constraint_data Indexed constraint data
     * @param validation_mode Connection validation mode
     * @param max_solutions Maximum number of solutions to collect
     * @param all_mappings_out Output vector populated with each solution (mapping[target_idx] = global_idx)
     * @param quiet_mode If true, suppress verbose info-level log messages
     * @param unique_shapes If true, solutions are unique by image set of global indices (see solve_topology_mapping_n)
     * @param initial_forbidden_shape_keys Sorted shape keys (global index tuples) treated as already used for
     *        uniqueness (e.g. exclusions from TopologyMappingEnumerationSession)
     * @return true if at least one solution was found
     */
    bool search_n(
        const GraphIndexData<TargetNode, GlobalNode>& graph_data,
        const ConstraintIndexData<TargetNode, GlobalNode>& constraint_data,
        ConnectionValidationMode validation_mode,
        size_t max_solutions,
        std::vector<std::vector<int>>& all_mappings_out,
        bool quiet_mode = false,
        bool unique_shapes = false,
        const std::vector<std::vector<int>>& initial_forbidden_shape_keys = {});

    /**
     * @brief Get the current search state
     *
     * @return const reference to the internal search state
     */
    const TopologySearchState& get_state() const { return state_; }

private:
    TopologySearchState state_;  // Internal state for the search
    bool quiet_mode_ = false;  // Quiet mode flag to suppress verbose debug messages
    /**
     * @brief Hash state for memoization (FNV-1a hash)
     *
     * @param mapping Current partial mapping
     * @return Hash value for the state
     */
    uint64_t hash_state(const std::vector<int>& mapping) const;

    /**
     * @brief Recursive DFS search
     *
     * @param pos Current position (number of assigned nodes)
     * @param graph_data Indexed graph data
     * @param constraint_data Indexed constraint data
     * @param validation_mode Connection validation mode
     * @return true if mapping found, false otherwise
     */
    bool dfs_recursive(
        size_t pos,
        const GraphIndexData<TargetNode, GlobalNode>& graph_data,
        const ConstraintIndexData<TargetNode, GlobalNode>& constraint_data,
        ConnectionValidationMode validation_mode);
};

/**
 * @brief SAT (CaDiCaL) search engine using hard CNF encoding plus preferred-hit maximization
 *
 * Encodes domain, degree, injectivity, edge preservation, same-rank groups, and cardinality, then searches for a
 * model that **maximizes the number of targets** whose chosen global lies in that target's preferred set (same notion
 * as `ConstraintIndexData::compute_constraint_stats` for `preferred_satisfied`). This uses auxiliary indicator
 * literals and repeated solves with an at-least-k cardinality over those indicators (small instance cap). When the
 * cap is exceeded or cardinality encoding is too large, falls back to a single satisfiability solve without that
 * objective. DFS still returns the **first** complete feasible mapping under its heuristic order, which can satisfy
 * strictly fewer preferred targets on the same instance.
 *
 * Channel/STRICT checks are still applied by MappingValidator after decode.
 *
 * In RELAXED mode, after locking the preferred-hit count (when that optimization runs), a second pass maximizes
 * auxiliary literals for per-edge channel thresholds so the embedding maximizes the same sum as DFS's relaxed
 * channel ordering objective (sum of min(required, actual) over target edges). When the number of threshold
 * literals exceeds a small cap, that k-descent pass is skipped (one final satisfiability solve still returns a valid
 * embedding). Other caps may also skip encoding or cardinality on very large instances.
 */
template <typename TargetNode, typename GlobalNode>
class SatSearchEngine {
public:
    bool search(
        const GraphIndexData<TargetNode, GlobalNode>& graph_data,
        const ConstraintIndexData<TargetNode, GlobalNode>& constraint_data,
        ConnectionValidationMode validation_mode,
        bool quiet_mode = false);

    /**
     * @brief Search for up to max_solutions distinct complete mappings using SAT with blocking clauses.
     *
     * After each SAT solve that returns SAT, the current assignment is decoded and pushed to
     * all_mappings_out. A blocking clause is then added — exact assignment, or a shape clause over the image set
     * when unique_shapes is true — and the solver is called again. This repeats until UNSAT or
     * all_mappings_out.size() >= max_solutions.
     *
     * @param graph_data Indexed graph data
     * @param constraint_data Indexed constraint data
     * @param validation_mode Connection validation mode
     * @param max_solutions Maximum number of solutions to collect
     * @param all_mappings_out Output vector populated with each solution (mapping[target_idx] = global_idx)
     * @param quiet_mode If true, suppress verbose info-level log messages
     * @param unique_shapes If true, block entire image-set equivalence classes per model (see solve_topology_mapping_n)
     * @param initial_forbidden_shape_keys Up-front shape keys to forbid (decoded with each fresh encoding)
     * @return true if at least one solution was found
     */
    bool search_n(
        const GraphIndexData<TargetNode, GlobalNode>& graph_data,
        const ConstraintIndexData<TargetNode, GlobalNode>& constraint_data,
        ConnectionValidationMode validation_mode,
        size_t max_solutions,
        std::vector<std::vector<int>>& all_mappings_out,
        bool quiet_mode = false,
        bool unique_shapes = false,
        const std::vector<std::vector<int>>& initial_forbidden_shape_keys = {});

    const TopologySearchState& get_state() const { return state_; }

private:
    TopologySearchState state_;
    bool quiet_mode_ = false;
};

/**
 * @brief Validates mappings and builds MappingResult
 *
 * Validates complete mappings, checks connection counts according to validation mode,
 * and builds MappingResult with detailed error messages and warnings.
 */
template <typename TargetNode, typename GlobalNode>
struct MappingValidator {
    /**
     * @brief Validate a complete mapping
     *
     * Checks that all edges exist, validates connection counts according to validation_mode,
     * and checks cardinality constraints.
     * In STRICT mode: fails if channel counts insufficient.
     * In RELAXED mode: collects warnings for insufficient channel counts but doesn't fail.
     * In NONE mode: skips channel count validation.
     *
     * @param mapping Complete mapping (mapping[i] = global_idx)
     * @param graph_data Indexed graph data
     * @param constraint_data Indexed constraint data (for cardinality constraint checking)
     * @param validation_mode Connection validation mode
     * @param warnings Optional vector to collect warnings
     * @param quiet_mode If true, log errors at debug level instead of error level
     * @return true if mapping is valid, false otherwise
     */
    static bool validate_mapping(
        const std::vector<int>& mapping,
        const GraphIndexData<TargetNode, GlobalNode>& graph_data,
        const ConstraintIndexData<TargetNode, GlobalNode>& constraint_data,
        ConnectionValidationMode validation_mode,
        std::vector<std::string>* warnings = nullptr,
        bool quiet_mode = false);

    /**
     * @brief Validate connection counts for all edges
     *
     * Checks channel counts and collects detailed warnings/errors according to validation_mode.
     *
     * @param mapping Complete mapping
     * @param graph_data Indexed graph data
     * @param validation_mode Connection validation mode
     * @param warnings Vector to collect warnings (must not be nullptr)
     * @param quiet_mode If true, log errors at debug level instead of error level
     */
    static void validate_connection_counts(
        const std::vector<int>& mapping,
        const GraphIndexData<TargetNode, GlobalNode>& graph_data,
        ConnectionValidationMode validation_mode,
        std::vector<std::string>* warnings,
        bool quiet_mode = false);

    /**
     * @brief Print mapping for debugging
     *
     * Prints the current mapping showing which target nodes map to which global nodes.
     * Useful for debugging mapping failures.
     *
     * @param mapping Mapping vector (mapping[i] = global_idx or -1)
     * @param graph_data Indexed graph data
     */
    static void print_mapping(
        const std::vector<int>& mapping, const GraphIndexData<TargetNode, GlobalNode>& graph_data);

    /**
     * @brief Build MappingResult from search state and mapping
     *
     * Converts internal mapping representation to MappingResult, computes constraint
     * statistics, and collects warnings. Provides detailed error messages when validation fails.
     *
     * **Important**: Even if validation fails, the closest/best mapping found is still saved
     * in the result. This allows users to see what progress was made and diagnose issues.
     * The `success` field indicates whether the mapping is valid, but `target_to_global`
     * and `global_to_target` will always contain the best mapping found (if any).
     *
     * @param mapping Complete mapping (may be incomplete if search failed)
     * @param graph_data Indexed graph data
     * @param constraint_data Indexed constraint data (for computing constraint stats)
     * @param state Search state (for statistics and error messages)
     * @param validation_mode Connection validation mode
     * @return MappingResult with success status, mappings (always saved even if invalid), warnings, and statistics
     */
    static MappingResult<TargetNode, GlobalNode> build_result(
        const std::vector<int>& mapping,
        const GraphIndexData<TargetNode, GlobalNode>& graph_data,
        const ConstraintIndexData<TargetNode, GlobalNode>& constraint_data,
        const TopologySearchState& state,
        ConnectionValidationMode validation_mode,
        bool quiet_mode = false);
};

}  // namespace detail

/**
 * @brief Incremental enumeration: each next() finds one mapping not listed in excluded_mappings.
 *
 * SAT reuses one CaDiCaL instance for a fixed graph/constraints/engine context: hard CNF is encoded once
 * (see sat_hard_constraint_encode_calls()), then each next() appends blocking clauses and solves again.
 *
 * DFS does **not** reuse search state across next() calls today: each call builds a new DFSSearchEngine and runs
 * search_n(..., excluded.size()+1, ...) from scratch, then returns the first mapping not in excluded_mappings.
 * That rediscovers earlier solutions internally and is why incremental DFS is often much slower than incremental
 * SAT on the same instance.
 *
 * **Possible future optimization:** a persistent DFS enumerator could resume after emitting each complete mapping
 * (e.g. iterative DFS with an explicit stack and “yield” at leaves, or a coroutine), while augmenting a growing set
 * of forbidden full assignments—similar amortization to SAT’s incremental blocking. Not implemented yet.
 */
template <typename TargetNode, typename GlobalNode>
class TopologyMappingEnumerationSession {
public:
    TopologyMappingEnumerationSession() = default;
    TopologyMappingEnumerationSession(const TopologyMappingEnumerationSession&) = delete;
    TopologyMappingEnumerationSession& operator=(const TopologyMappingEnumerationSession&) = delete;
    TopologyMappingEnumerationSession(TopologyMappingEnumerationSession&&) noexcept = default;
    TopologyMappingEnumerationSession& operator=(TopologyMappingEnumerationSession&&) noexcept = default;
    ~TopologyMappingEnumerationSession();

    void reset() noexcept;

    MappingResult<TargetNode, GlobalNode> next(
        const AdjacencyGraph<TargetNode>& target_graph,
        const AdjacencyGraph<GlobalNode>& global_graph,
        const MappingConstraints<TargetNode, GlobalNode>& constraints,
        const std::vector<std::map<TargetNode, GlobalNode>>& excluded_mappings,
        ConnectionValidationMode connection_validation_mode = ConnectionValidationMode::RELAXED,
        bool quiet_mode = false,
        TopologyMappingSolverEngine solver_engine = TopologyMappingSolverEngine::Auto,
        bool unique_shapes = false);

    size_t sat_solve_calls() const noexcept { return sat_solve_calls_; }

    /** SAT only: number of successful hard-constraint CNF encodings in this session (0 if using DFS). */
    size_t sat_hard_constraint_encode_calls() const noexcept { return sat_hard_constraint_encode_calls_; }

private:
    bool ready_{false};
    bool quiet_{false};
    bool unique_shapes_{false};
    bool use_sat_{false};
    size_t sat_exclusions_encoded_{0};
    size_t sat_solve_calls_{0};
    size_t sat_hard_constraint_encode_calls_{0};
    AdjacencyGraph<TargetNode> snap_target_{};
    AdjacencyGraph<GlobalNode> snap_global_{};
    TopologyMappingSolverEngine engine_{TopologyMappingSolverEngine::Auto};
    ConnectionValidationMode mode_{ConnectionValidationMode::RELAXED};
    std::optional<detail::GraphIndexData<TargetNode, GlobalNode>> graph_data_;
    std::optional<detail::ConstraintIndexData<TargetNode, GlobalNode>> constraint_data_;
    std::unique_ptr<detail::TopologySatSession, detail::TopologySatSessionDeleter> sat_session_{};
    detail::TopologySatHardEncoding sat_enc_{};
};

}  // namespace tt::tt_fabric

// Include template implementations
// Define guard macro to prevent circular include when .tpp includes this header
#ifndef TOPOLOGY_SOLVER_TPP_INCLUDING
#define TOPOLOGY_SOLVER_TPP_INCLUDING
#endif
// NOLINTNEXTLINE(misc-header-include-cycle) - Guard macro prevents actual circular dependency
#include <tt-metalium/experimental/fabric/topology_solver.tpp>
