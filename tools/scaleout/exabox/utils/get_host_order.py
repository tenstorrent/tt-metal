#!/usr/bin/env python3
"""
This script parses a CSV cutsheet or textproto cabling_descriptor file to extract the optimal host order for fabric tests.

Parses CSV or textproto descriptor files to extract hostname, aisle, and rack information.
Builds an interconnected node graph. 
Finds a Hamiltonian cycle favoring near rack hops.

Returns a comma-separated list of hostnames in the optimal order for testing, which can be used to set the HOSTS environment variable for fabric tests.
"""

import csv
import re
import sys
from collections import defaultdict, namedtuple
from typing import Dict, List, Set, Tuple, Optional

# Node information structure
NodeInfo = namedtuple('NodeInfo', ['hostname', 'aisle', 'rack'])

class HamiltonianCycleFinder:
    def __init__(self):
        self.nodes: Dict[str, NodeInfo] = {}
        self.graph: Dict[str, Set[str]] = defaultdict(set)
    
    def parse_csv(self, csv_file: str) -> None:
        """Parse CSV file and extract connections between different hosts."""
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            # Skip header row
            next(reader)
            
            for row in reader:
                if len(row) < 13:  # Ensure we have enough columns
                    continue
                
                # Extract source and destination info
                src_hostname = row[0].strip()
                src_aisle = row[2].strip()
                src_rack = int(row[3]) if row[3].strip().isdigit() else 0
                
                dst_hostname = row[9].strip()
                dst_aisle = row[11].strip()
                dst_rack = int(row[12]) if row[12].strip().isdigit() else 0
                
                # Skip empty rows or intra-host connections
                if not src_hostname or not dst_hostname or src_hostname == dst_hostname:
                    continue
                
                # Store node information
                self.nodes[src_hostname] = NodeInfo(src_hostname, src_aisle, src_rack)
                self.nodes[dst_hostname] = NodeInfo(dst_hostname, dst_aisle, dst_rack)
                
                # Add bidirectional connection
                self.graph[src_hostname].add(dst_hostname)
                self.graph[dst_hostname].add(src_hostname)
    
    def parse_textproto(self, textproto_file: str) -> None:
        """Parse textproto file and extract connections between different hosts."""
        with open(textproto_file, 'r') as f:
            content = f.read()
        
        # Extract all hostnames from children sections
        children_pattern = r'children\s*{\s*name:\s*"([^"]+)"'
        hostnames = re.findall(children_pattern, content)
        
        # Parse hostname to get aisle and rack info
        for hostname in hostnames:
            aisle, rack = self._parse_hostname(hostname)
            self.nodes[hostname] = NodeInfo(hostname, aisle, rack)
        
        # Extract connections from the textproto
        connections_pattern = r'connections\s*{\s*port_a\s*{\s*path:\s*"([^"]+)"[^}]+}\s*port_b\s*{\s*path:\s*"([^"]+)"[^}]+}\s*}'
        connections = re.findall(connections_pattern, content)
        
        # Process connections and filter out intra-host connections
        for src_hostname, dst_hostname in connections:
            # Skip intra-host connections (same hostname)
            if src_hostname == dst_hostname:
                continue
            
            # Only add connections between hosts we've identified
            if src_hostname in self.nodes and dst_hostname in self.nodes:
                self.graph[src_hostname].add(dst_hostname)
                self.graph[dst_hostname].add(src_hostname)
    
    def _parse_hostname(self, hostname: str) -> Tuple[str, int]:
        """Parse hostname to extract aisle and rack information.
        
        Expected format: bh-glx-c01u02 where 'c' is aisle and '01' is rack
        """
        # Pattern to match hostname format like "bh-glx-c01u02"
        pattern = r'bh-glx-([a-z])([0-9]+)u([0-9]+)'
        match = re.match(pattern, hostname)
        
        if match:
            aisle = match.group(1)  # 'c' or 'd'
            rack = int(match.group(2))  # '01' -> 1
            return aisle, rack
        else:
            # Fallback for unexpected format
            return 'unknown', 0
    
    def filter_to_hosts(self, host_list: List[str]) -> None:
        """Filter the graph to only include specified hosts and their connections."""
        # Convert to set for faster lookup
        allowed_hosts = set(host_list)
        
        # Remove nodes not in the allowed list
        nodes_to_remove = [node for node in self.nodes.keys() if node not in allowed_hosts]
        for node in nodes_to_remove:
            del self.nodes[node]
            if node in self.graph:
                del self.graph[node]
        
        # Remove connections to hosts not in the allowed list
        for node in self.graph:
            self.graph[node] = {neighbor for neighbor in self.graph[node] if neighbor in allowed_hosts}
        
        # Remove any nodes that have no connections after filtering
        disconnected_nodes = [node for node, connections in self.graph.items() if not connections]
        for node in disconnected_nodes:
            if node in self.nodes:
                del self.nodes[node]
            if node in self.graph:
                del self.graph[node]
    
    def calculate_distance(self, node1: str, node2: str) -> float:
        """Calculate distance metric favoring same aisle and nearby racks."""
        info1 = self.nodes[node1]
        info2 = self.nodes[node2]
        
        # Same aisle gets priority (lower distance)
        if info1.aisle == info2.aisle:
            rack_diff = abs(info1.rack - info2.rack)
            return rack_diff
        else:
            # Different aisles get higher distance
            aisle_penalty = 100
            rack_diff = abs(info1.rack - info2.rack)
            return aisle_penalty + rack_diff
    
    def get_candidate_starting_nodes(self) -> List[str]:
        """Get candidate starting nodes, prioritizing lowest aisle/rack with good connectivity.
        
        Falls back to degree-then-name ordering when physical location data is unavailable.
        """
        # Group nodes by aisle and rack
        aisle_nodes = defaultdict(list)
        for hostname, info in self.nodes.items():
            aisle_nodes[info.aisle].append((info.rack, hostname))
        
        # If no physical location data is available, sort by connectivity degree then name.
        if list(aisle_nodes.keys()) == ['unknown']:
            return sorted(
                [h for h, _ in aisle_nodes['unknown'] if len(self.graph[h]) >= 2],
                key=lambda h: (-len(self.graph[h]), h)
            )
        
        candidates = []
        
        # For each aisle (sorted), get nodes from lowest to highest rack
        for aisle in sorted(aisle_nodes.keys()):
            rack_nodes = sorted(aisle_nodes[aisle])
            for rack, hostname in rack_nodes:
                # Only consider nodes with reasonable connectivity
                if len(self.graph[hostname]) >= 2:
                    candidates.append(hostname)
        
        return candidates
    
    def get_aisle_aware_neighbors(self, node: str, visited: Set[str]) -> List[str]:
        """Get unvisited neighbors with aisle-completion priority.
        
        Falls back to connectivity-only ordering (sorted by name for determinism)
        when the node has no parseable physical location data.
        """
        unvisited_neighbors = [n for n in self.graph[node] if n not in visited]
        
        if not unvisited_neighbors:
            return []
        
        current_info = self.nodes[node]
        
        # If physical location is unavailable, skip topology heuristics entirely.
        if current_info.aisle == 'unknown':
            return sorted(unvisited_neighbors)
        
        same_rack = []
        same_aisle_near = []
        same_aisle_far = []
        cross_aisle = []
        
        for neighbor in unvisited_neighbors:
            neighbor_info = self.nodes[neighbor]
            
            if neighbor_info.aisle == current_info.aisle and neighbor_info.rack == current_info.rack:
                same_rack.append(neighbor)
            elif neighbor_info.aisle == current_info.aisle:
                rack_diff = abs(neighbor_info.rack - current_info.rack)
                if rack_diff <= 2:
                    same_aisle_near.append((rack_diff, neighbor))
                else:
                    same_aisle_far.append((rack_diff, neighbor))
            else:
                distance = self.calculate_distance(node, neighbor)
                cross_aisle.append((distance, neighbor))
        
        # Sort each group and combine
        result = []
        
        # Same rack neighbors first (no specific order needed)
        result.extend(same_rack)
        
        # Same aisle, nearby racks (sorted by distance)
        same_aisle_near.sort(key=lambda x: x[0])
        result.extend([neighbor for _, neighbor in same_aisle_near])
        
        # Same aisle, far racks (sorted by distance)
        same_aisle_far.sort(key=lambda x: x[0])
        result.extend([neighbor for _, neighbor in same_aisle_far])
        
        # Cross-aisle (sorted by distance)
        cross_aisle.sort(key=lambda x: x[0])
        result.extend([neighbor for _, neighbor in cross_aisle])
        
        return result
    
    def count_remaining_in_aisle(self, aisle: str, visited: Set[str]) -> int:
        """Count how many unvisited nodes remain in the given aisle."""
        count = 0
        for hostname, info in self.nodes.items():
            if info.aisle == aisle and hostname not in visited:
                count += 1
        return count
    
    def find_hamiltonian_cycle(self, start_node: Optional[str] = None) -> Optional[List[str]]:
        """Find Hamiltonian cycle with improved aisle-aware algorithm."""
        if not self.nodes:
            return None
        
        # If no start node specified, try multiple candidates
        if start_node is None:
            candidates = self.get_candidate_starting_nodes()
            
            best_cycle = None
            best_distance = float('inf')
            
            for i, candidate in enumerate(candidates[:10]):  # Try up to 10 candidates
                cycle = self._find_cycle_from_start(candidate)
                if cycle:
                    # Calculate total distance for this cycle
                    total_distance = sum(
                        self.calculate_distance(cycle[j], cycle[(j + 1) % len(cycle)])
                        for j in range(len(cycle))
                    )
                    
                    if total_distance < best_distance:
                        best_distance = total_distance
                        best_cycle = cycle
            
            return best_cycle
        else:
            return self._find_cycle_from_start(start_node)
    
    def _find_cycle_from_start(self, start_node: str) -> Optional[List[str]]:
        """Find Hamiltonian cycle from a specific starting node."""
        def backtrack(path: List[str], visited: Set[str]) -> Optional[List[str]]:
            current = path[-1]
            
            # If we've visited all nodes, check if we can return to start
            if len(visited) == len(self.nodes):
                if start_node in self.graph[current]:
                    return path + [start_node]
                else:
                    return None
            
            # Use aisle-aware neighbor selection
            neighbors = self.get_aisle_aware_neighbors(current, visited)
            
            # Additional heuristic: if we're in the middle of an aisle with many unvisited nodes,
            # strongly prefer staying in the same aisle.
            # Only applied when physical location data is available.
            current_info = self.nodes[current]
            if current_info.aisle != 'unknown':
                remaining_in_aisle = self.count_remaining_in_aisle(current_info.aisle, visited)
                
                if remaining_in_aisle > 3:  # If many nodes left in current aisle
                    # Filter to prioritize same-aisle neighbors even more
                    same_aisle_neighbors = [
                        n for n in neighbors 
                        if self.nodes[n].aisle == current_info.aisle
                    ]
                    if same_aisle_neighbors:
                        neighbors = same_aisle_neighbors + [
                            n for n in neighbors 
                            if self.nodes[n].aisle != current_info.aisle
                        ]
            
            for neighbor in neighbors:
                visited.add(neighbor)
                path.append(neighbor)
                
                result = backtrack(path, visited)
                if result is not None:
                    return result
                
                # Backtrack
                path.pop()
                visited.remove(neighbor)
            
            return None
        
        visited = {start_node}
        path = [start_node]
        
        cycle = backtrack(path, visited)
        return cycle[:-1] if cycle else None  # Remove duplicate start node at end
    
    def print_graph_info(self) -> None:
        """Print graph information for debugging."""
        print(f"Total nodes: {len(self.nodes)}")
        print(f"Nodes by aisle:")
        
        aisles = defaultdict(list)
        for node_info in self.nodes.values():
            aisles[node_info.aisle].append(f"{node_info.hostname}(R{node_info.rack:02d})")
        
        for aisle in sorted(aisles.keys()):
            print(f"  Aisle {aisle}: {', '.join(sorted(aisles[aisle]))}")
        
        print(f"\nConnectivity:")
        for node in sorted(self.nodes.keys()):
            print(f"  {node}: {len(self.graph[node])} connections")

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python hamiltonian_cycle_finder.py <input_file> [host_list]")
        print("  input_file: Path to the CSV or textproto descriptor file")
        print("  host_list: Optional comma-separated list of hostnames to include in the cycle")
        sys.exit(1)
    
    input_file = sys.argv[1]
    host_filter = sys.argv[2] if len(sys.argv) == 3 else None
    
    try:
        finder = HamiltonianCycleFinder()
        
        # Determine file format and parse accordingly
        if input_file.endswith('.textproto'):
            finder.parse_textproto(input_file)
        else:
            # Default to CSV parsing
            finder.parse_csv(input_file)
        
        if not finder.nodes:
            print("No valid connections found in the input file.")
            sys.exit(1)
        
        # Apply host filtering if provided
        if host_filter:
            host_list = [host.strip() for host in host_filter.split(',') if host.strip()]
            if not host_list:
                print("Error: Empty host list provided.")
                sys.exit(1)
            
            # Check if all specified hosts exist in the file
            missing_hosts = [host for host in host_list if host not in finder.nodes]
            if missing_hosts:
                print(f"Error: The following hosts were not found in the input file: {', '.join(missing_hosts)}")
                sys.exit(1)
            
            finder.filter_to_hosts(host_list)
            
            if not finder.nodes:
                print("No valid connections found among the specified hosts.")
                sys.exit(1)
        
        # Find Hamiltonian cycle
        cycle = finder.find_hamiltonian_cycle()
        
        if cycle:
            print(','.join(cycle))
        else:
            print("No Hamiltonian cycle found.")
            sys.exit(1)
            
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()