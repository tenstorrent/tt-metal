// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
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
     * @brief Print adjacency map for debugging
     *
     * Prints the graph structure showing each node and its neighbors.
     * Useful for debugging mapping failures.
     *
     * @param graph_name Name to identify this graph in the output
     */
    void print_adjacency_map(const std::string& graph_name = "Graph") const;

private:
    AdjacencyMap adj_map_;
    std::vector<NodeId> nodes_cache_;
};

std::map<MeshId, AdjacencyGraph<FabricNodeId>> build_adjacency_graph_logical(const MeshGraph& mesh_graph);

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
     * Throws TT_THROW if constraint causes conflicts (empty valid mappings).
     *
     * @tparam TraitType The type of the trait value (must be explicitly specified)
     * @param target_traits Map from target nodes to their trait values
     * @param global_traits Map from global nodes to their trait values
     * @throws std::runtime_error If constraint causes empty valid mappings for any target node
     */
    template <typename TraitType>
    void add_required_trait_constraint(
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
     * @throws std::runtime_error If constraint causes empty valid mappings
     */
    void add_required_constraint(TargetNode target_node, GlobalNode global_node);

    /**
     * @brief Add explicit required constraint (one-to-many for target node)
     *
     * Constrains a specific target node to map to any of the provided global nodes.
     * Intersects with existing constraints. Throws TT_THROW if constraint causes conflicts.
     *
     * @param target_node The target node to constrain
     * @param global_nodes The set of global nodes it can map to
     * @throws std::runtime_error If constraint causes empty valid mappings
     */
    void add_required_constraint(TargetNode target_node, const std::set<GlobalNode>& global_nodes);

    /**
     * @brief Add explicit required constraint (one-to-many for global node)
     *
     * Constrains multiple target nodes to map to a specific global node.
     * Intersects with existing constraints. Throws TT_THROW if constraint causes conflicts.
     *
     * @param target_nodes The set of target nodes to constrain
     * @param global_node The global node they must map to
     * @throws std::runtime_error If constraint causes empty valid mappings
     */
    void add_required_constraint(const std::set<TargetNode>& target_nodes, GlobalNode global_node);

    /**
     * @brief Add explicit required constraint (many-to-many)
     *
     * Constrains multiple target nodes to map to any of the provided global nodes.
     * This creates a many-to-many relationship: any target node from the set can map
     * to any global node from the set. Intersects with existing constraints for each target.
     * Throws TT_THROW if constraint causes conflicts.
     *
     * @param target_nodes The set of target nodes to constrain
     * @param global_nodes The set of global nodes they can map to
     * @throws std::runtime_error If constraint causes empty valid mappings
     */
    void add_required_constraint(const std::set<TargetNode>& target_nodes, const std::set<GlobalNode>& global_nodes);

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
     * Removes the mapping from valid mappings. Throws TT_THROW if constraint contradicts
     * a required constraint or causes empty valid mappings.
     *
     * @param target_node The target node to constrain
     * @param global_node The global node it cannot map to
     * @throws std::runtime_error If constraint contradicts required constraint or causes empty valid mappings
     */
    void add_forbidden_constraint(TargetNode target_node, GlobalNode global_node);

    /**
     * @brief Add explicit forbidden constraint (one-to-many for target node)
     *
     * Forbids a specific target node from mapping to any of the provided global nodes.
     * Removes the mappings from valid mappings. Throws TT_THROW if constraint contradicts
     * a required constraint or causes empty valid mappings.
     *
     * @param target_node The target node to constrain
     * @param global_nodes The set of global nodes it cannot map to
     * @throws std::runtime_error If constraint contradicts required constraint or causes empty valid mappings
     */
    void add_forbidden_constraint(TargetNode target_node, const std::set<GlobalNode>& global_nodes);

    /**
     * @brief Add explicit forbidden constraint (one-to-many for global node)
     *
     * Forbids multiple target nodes from mapping to a specific global node.
     * Removes the mappings from valid mappings. Throws TT_THROW if constraint contradicts
     * a required constraint or causes empty valid mappings.
     *
     * @param target_nodes The set of target nodes to constrain
     * @param global_node The global node they cannot map to
     * @throws std::runtime_error If constraint contradicts required constraint or causes empty valid mappings
     */
    void add_forbidden_constraint(const std::set<TargetNode>& target_nodes, GlobalNode global_node);

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
     * @throws std::runtime_error If min_count is greater than the number of pairs
     */
    void add_cardinality_constraint(
        const std::set<std::pair<TargetNode, GlobalNode>>& mapping_pairs, size_t min_count = 1);

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
     * @throws std::runtime_error If min_count is greater than the number of possible pairs
     */
    void add_cardinality_constraint(
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
     * @return const std::vector<std::pair<std::set<std::pair<TargetNode, GlobalNode>>, size_t>>&
     *         Vector of (mapping_pairs, min_count) tuples representing cardinality constraints
     */
    const std::vector<std::pair<std::set<std::pair<TargetNode, GlobalNode>>, size_t>>& get_cardinality_constraints()
        const;

private:
    // Internal representation: intersection of all constraints
    std::map<TargetNode, std::set<GlobalNode>> valid_mappings_;      // Required constraints
    std::map<TargetNode, std::set<GlobalNode>> preferred_mappings_;  // Preferred constraints

    // Cardinality constraints: vector of (mapping_pairs, min_count) tuples
    // Each constraint requires that at least min_count of the mapping_pairs must be satisfied
    std::vector<std::pair<std::set<std::pair<TargetNode, GlobalNode>>, size_t>> cardinality_constraints_;

    // Track which global nodes are exclusively reserved by many-to-many constraints
    // Maps global node -> set of target nodes that are allowed to map to it via many-to-many constraints
    std::map<GlobalNode, std::set<TargetNode>> reserved_global_nodes_;

    // Helper to intersect two sets
    static std::set<GlobalNode> intersect_sets(const std::set<GlobalNode>& set1, const std::set<GlobalNode>& set2);

    // Internal validation - throws if invalid
    // If saved_state is provided and validation fails, restores the saved state before throwing
    void validate_and_throw(const std::map<TargetNode, std::set<GlobalNode>>* saved_state = nullptr);

    // Validate that all cardinality constraints are compatible with required constraints
    // and that they are satisfiable together
    void validate_cardinality_constraints() const;
};

/**
 * @brief Mode for connection count validation
 */
enum class ConnectionValidationMode {
    STRICT,  ///< Strict mode: require exact channel counts, fail if not met
    RELAXED  ///< Relaxed mode: prefer correct channel counts, but allow mismatches with warnings (default)
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
 * then optimizes for preferred constraints.
 *
 * @tparam TargetNode The type used to identify nodes in the target graph (must be explicitly specified)
 * @tparam GlobalNode The type used to identify nodes in the global graph (must be explicitly specified)
 * @param target_graph The target graph (subgraph pattern to find)
 * @param global_graph The global graph (larger host graph that contains the target)
 * @param constraints The mapping constraints to satisfy
 * @param connection_validation_mode How to validate connection counts (default: RELAXED)
 * @param quiet_mode If true, log errors at debug level instead of error level (useful for auto-discovery)
 * @return MappingResult containing success status, bidirectional mappings, and warnings
 */
template <typename TargetNode, typename GlobalNode>
MappingResult<TargetNode, GlobalNode> solve_topology_mapping(
    const AdjacencyGraph<TargetNode>& target_graph,
    const AdjacencyGraph<GlobalNode>& global_graph,
    const MappingConstraints<TargetNode, GlobalNode>& constraints,
    ConnectionValidationMode connection_validation_mode = ConnectionValidationMode::RELAXED,
    bool quiet_mode = false);

namespace detail {

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
    // conn_count[i][j] = number of channels from node i to node j
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

    // Preferred mappings: target_idx -> vector of preferred global_indices
    // Used for optimization, doesn't restrict valid mappings
    std::vector<std::vector<size_t>> preferred_global_indices;

    // Cardinality constraints: vector of (mapping_pairs_as_indices, min_count) tuples
    // Each constraint requires that at least min_count of the (target_idx, global_idx) pairs must be satisfied
    std::vector<std::pair<std::set<std::pair<size_t, size_t>>, size_t>> cardinality_constraints;

    /**
     * @brief Construct ConstraintIndexData from MappingConstraints and GraphIndexData
     *
     * Builds indexed constraint representation from constraints and graph data.
     */
    ConstraintIndexData(
        const MappingConstraints<TargetNode, GlobalNode>& constraints,
        const GraphIndexData<TargetNode, GlobalNode>& graph_data);

    // Helper: check if mapping is valid
    bool is_valid_mapping(size_t target_idx, size_t global_idx) const;

    // Helper: get candidates for target node
    // Returns restricted candidates if available, otherwise returns empty vector (meaning all are valid)
    const std::vector<size_t>& get_candidates(size_t target_idx) const;

    // Helper: check if cardinality constraints are satisfied by a complete mapping
    // Returns true if all cardinality constraints are satisfied, false otherwise
    bool check_cardinality_constraints(const std::vector<int>& mapping) const;

    // Helper: check if cardinality constraints can still be satisfied by a partial mapping
    // Returns true if it's still possible to satisfy all cardinality constraints, false if impossible
    bool can_satisfy_cardinality_constraints(const std::vector<int>& mapping) const;

    // Helper: check if a target node has exactly one required constraint (for pre-assignment)
    // Returns the global index if exactly one, SIZE_MAX otherwise
    size_t get_single_required_mapping(size_t target_idx) const;

    // Helper: compute constraint statistics from a complete mapping
    // Returns (required_satisfied, preferred_satisfied, preferred_total)
    std::tuple<size_t, size_t, size_t> compute_constraint_stats(
        const std::vector<int>& mapping, const GraphIndexData<TargetNode, GlobalNode>& graph_data) const;
};

/**
 * @brief DFS search engine for topology mapping
 *
 * Implements backtracking search with memoization and consistency checking.
 * Uses SearchHeuristic for node selection and candidate generation.
 *
 * **Important**: Even if the search fails to find a complete valid mapping, the
 * `SearchState::mapping` will contain the best/closest partial mapping found.
 * This allows users to see what progress was made and diagnose why the search failed.
 * The MappingValidator will save this partial mapping in the result even if validation fails.
 */
template <typename TargetNode, typename GlobalNode>
class DFSSearchEngine {
public:
    /**
     * @brief Search state tracking mapping progress and statistics
     *
     * **Note**: The `mapping` vector always contains the best mapping found so far,
     * even if the search fails. This allows users to inspect partial mappings for debugging.
     */
    struct SearchState {
        std::vector<int> mapping;                    // mapping[target_idx] = global_idx or -1 (best found so far)
        std::vector<bool> used;                      // used[global_idx] = true if assigned
        std::unordered_set<uint64_t> failed_states;  // Memoization cache of failed states
        size_t dfs_calls = 0;                        // Number of DFS calls made
        size_t backtrack_count = 0;                  // Number of backtracks performed
        std::string error_message;                   // Error message if search fails
    };

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
     * @brief Get the current search state
     *
     * @return const reference to the internal search state
     */
    const SearchState& get_state() const { return state_; }

private:
    SearchState state_;  // Internal state for the search
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
        const DFSSearchEngine<TargetNode, GlobalNode>::SearchState& state,
        ConnectionValidationMode validation_mode,
        bool quiet_mode = false);
};

}  // namespace detail

}  // namespace tt::tt_fabric

// Include template implementations
#include <tt-metalium/experimental/fabric/topology_solver.tpp>
