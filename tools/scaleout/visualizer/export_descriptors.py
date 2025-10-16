#!/usr/bin/env python3
"""
Export Tool - Extract cabling topology and deployment information from cytoscape.js visualization
Combined version - contains both CablingDescriptor and DeploymentDescriptor export functionality
"""

import json
import argparse
import sys
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
import re

# Add the protobuf directory to Python path for protobuf imports
# Check for TT_METAL_HOME environment variable
tt_metal_home = os.environ.get("TT_METAL_HOME")
if not tt_metal_home:
    print("Error: TT_METAL_HOME environment variable is not set")
    print("Please set TT_METAL_HOME to the root directory of your tt-metal repository")
    sys.exit(1)

if not os.path.exists(tt_metal_home):
    print(f"Error: TT_METAL_HOME path does not exist: {tt_metal_home}")
    print("Please set TT_METAL_HOME to a valid directory")
    sys.exit(1)

protobuf_dir = os.path.join(tt_metal_home, "build", "tools", "scaleout", "protobuf")
sys.path.append(protobuf_dir)


try:
    import cluster_config_pb2
    import deployment_pb2
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print(f"Make sure cluster_config_pb2.py and deployment_pb2.py are available in {protobuf_dir}")
    print("This should be: $TT_METAL_HOME/build/tools/scaleout/protobuf/")
    sys.exit(1)

# Import protobuf modules
try:
    from google.protobuf import text_format
    from google.protobuf.message import Message
except ImportError:
    print("Warning: protobuf not available. Deployment descriptor export will not work.")
    text_format = None
    Message = None


class CytoscapeDataParser:
    """Parse Cytoscape.js data and extract connection information"""

    def __init__(self, data: Dict):
        self.data = data
        self.nodes = {}
        self.edges = []
        self._parse_data()

    def _parse_data(self):
        """Parse Cytoscape data into nodes and edges"""
        elements = self.data.get("elements", [])

        for element in elements:
            if "source" in element.get("data", {}):
                # This is an edge
                self.edges.append(element)
            else:
                # This is a node
                node_data = element.get("data", {})
                node_id = node_data.get("id")
                if node_id:
                    self.nodes[node_id] = element

    def extract_hierarchy_info(self, node_id: str) -> Optional[Dict]:
        """Extract shelf/tray/port info from node ID using only patterns that are actually used"""

        # Define patterns with their handlers - only include patterns that are actually used
        # Order matters: more specific patterns first, fallback last
        patterns = [
            # Current standard: <label>-tray#-port# (used by both 20-column and 8-column CSV formats)
            (r"(.+)-tray(\d+)-port(\d+)", self._handle_preferred_port),
            (r"(.+)-tray(\d+)", self._handle_preferred_tray),
            # Legacy ID pattern: port_<hostname>_<tray>_<port> (old node ID convention)
            (r"port_(.+)_(\d+)_(\d+)", self._handle_hostname_port),
            (r"tray_(.+)_(\d+)", self._handle_hostname_tray),
            (r"shelf_(.+)", self._handle_hostname_shelf),
            # Legacy ID pattern: port_<rack>_U<shelf>_<tray>_<port> (old node ID convention)
            (r"port_(\d+)_U(\d+)_(\d+)_(\d+)", self._handle_rack_hierarchy_port),
            (r"tray_(\d+)_U(\d+)_(\d+)", self._handle_rack_hierarchy_tray),
            (r"shelf_(\d+)_U(\d+)", self._handle_rack_hierarchy_shelf),
            # Fallback for any other format
            (r"(.+)", self._handle_preferred_shelf),
        ]

        for pattern, handler in patterns:
            match = re.match(pattern, node_id)
            if match:
                return handler(match.groups())

        return None

    # Pattern handlers for node ID formats (current standard and legacy patterns)
    def _handle_preferred_port(self, groups):
        """Handle <label>-tray#-port# format (current standard node ID format)"""
        return {
            "type": "port",
            "hostname": groups[0],
            "shelf_id": groups[0],
            "tray_id": int(groups[1]),
            "port_id": int(groups[2]),
        }

    def _handle_preferred_tray(self, groups):
        """Handle <label>-tray# format (current standard node ID format)"""
        return {"type": "tray", "hostname": groups[0], "shelf_id": groups[0], "tray_id": int(groups[1])}

    def _handle_preferred_shelf(self, groups):
        """Handle <label> format (current standard node ID format, fallback for any unmatched ID)"""
        return {"type": "shelf", "hostname": groups[0], "shelf_id": groups[0]}

    def _handle_hostname_port(self, groups):
        """Handle port_<hostname>_<tray>_<port> format (legacy node ID pattern)"""
        hostname = groups[0]
        return {
            "type": "port",
            "hostname": hostname,
            "shelf_id": hostname,
            "tray_id": int(groups[1]),
            "port_id": int(groups[2]),
        }

    def _handle_hostname_tray(self, groups):
        """Handle tray_<hostname>_<tray> format (legacy node ID pattern)"""
        hostname = groups[0]
        return {"type": "tray", "hostname": hostname, "shelf_id": hostname, "tray_id": int(groups[1])}

    def _handle_hostname_shelf(self, groups):
        """Handle shelf_<hostname> format (legacy node ID pattern)"""
        hostname = groups[0]
        return {"type": "shelf", "hostname": hostname, "shelf_id": hostname}

    def _handle_rack_hierarchy_port(self, groups):
        """Handle port_<rack>_U<shelf>_<tray>_<port> format (legacy node ID pattern)"""
        shelf_id = f"{groups[0]}_U{groups[1]}"
        return {
            "type": "port",
            "hostname": shelf_id,
            "shelf_id": shelf_id,
            "tray_id": int(groups[2]),
            "port_id": int(groups[3]),
        }

    def _handle_rack_hierarchy_tray(self, groups):
        """Handle tray_<rack>_U<shelf>_<tray> format (legacy node ID pattern)"""
        shelf_id = f"{groups[0]}_U{groups[1]}"
        return {"type": "tray", "hostname": shelf_id, "shelf_id": shelf_id, "tray_id": int(groups[2])}

    def _handle_rack_hierarchy_shelf(self, groups):
        """Handle shelf_<rack>_U<shelf> format (legacy node ID pattern)"""
        shelf_id = f"{groups[0]}_U{groups[1]}"
        return {"type": "shelf", "hostname": shelf_id, "shelf_id": shelf_id}

    def extract_connections(self) -> List[Dict]:
        """Extract connection information from edges"""
        connections = []

        for edge in self.edges:
            edge_data = edge.get("data", {})
            source_id = edge_data.get("source")
            target_id = edge_data.get("target")

            if not source_id or not target_id:
                continue

            # Extract hierarchy info for both endpoints
            source_info = self.extract_hierarchy_info(source_id)
            target_info = self.extract_hierarchy_info(target_id)

            if not source_info or not target_info:
                continue

            # Only process port-to-port connections
            if source_info.get("type") == "port" and target_info.get("type") == "port":
                connection = {
                    "source": {
                        "hostname": source_info.get("hostname"),
                        "shelf_id": source_info.get("shelf_id"),
                        "tray_id": source_info.get("tray_id"),
                        "port_id": source_info.get("port_id"),
                    },
                    "target": {
                        "hostname": target_info.get("hostname"),
                        "shelf_id": target_info.get("shelf_id"),
                        "tray_id": target_info.get("tray_id"),
                        "port_id": target_info.get("port_id"),
                    },
                }
                connections.append(connection)

        return connections


class VisualizerCytoscapeDataParser(CytoscapeDataParser):
    """Parser for visualizer-specific Cytoscape data"""

    def extract_connections(self) -> List[Dict]:
        """Extract connection information from edges"""
        connections = []

        for edge in self.edges:
            edge_data = edge.get("data", {})
            source_id = edge_data.get("source")
            target_id = edge_data.get("target")

            if not source_id or not target_id:
                continue

            # Extract hierarchy info for both endpoints
            source_info = self.extract_hierarchy_info(source_id)
            target_info = self.extract_hierarchy_info(target_id)

            if not source_info or not target_info:
                continue

            # Only process port-to-port connections
            if source_info.get("type") == "port" and target_info.get("type") == "port":
                # Get hostname from multiple sources (priority order):
                # 1. Edge data (source_hostname/destination_hostname from 20-column format)
                # 2. Node data (from port/tray/shelf hierarchy)
                # 3. ID pattern matching (fallback)
                source_hostname = (
                    edge_data.get("source_hostname")
                    or self._get_hostname_from_port(source_id)
                    or source_info.get("hostname")
                )
                target_hostname = (
                    edge_data.get("destination_hostname")
                    or self._get_hostname_from_port(target_id)
                    or target_info.get("hostname")
                )

                # Get node_type from the shelf nodes
                source_node_type = self._get_node_type_from_port(source_id)
                target_node_type = self._get_node_type_from_port(target_id)

                connection = {
                    "source": {
                        "hostname": source_hostname,
                        "shelf_id": source_info.get("shelf_id"),
                        "tray_id": source_info.get("tray_id"),
                        "port_id": source_info.get("port_id"),
                        "node_type": source_node_type,
                    },
                    "target": {
                        "hostname": target_hostname,
                        "shelf_id": target_info.get("shelf_id"),
                        "tray_id": target_info.get("tray_id"),
                        "port_id": target_info.get("port_id"),
                        "node_type": target_node_type,
                    },
                }
                connections.append(connection)

        return connections

    def _get_hostname_from_port(self, port_id: str) -> Optional[str]:
        """Get hostname from a port node's data (for 20-column format)"""
        # Find the port node in the cytoscape data
        for element in self.data.get("elements", []):
            if element.get("data", {}).get("id") == port_id:
                node_data = element.get("data", {})
                # Check if hostname is stored directly in the port data
                hostname = node_data.get("hostname")
                if hostname and hostname.strip():
                    return hostname.strip()

                # If not in port data, traverse up to get from parent shelf
                parent_id = node_data.get("parent")
                if parent_id:
                    # Find the parent (tray) node
                    for parent_element in self.data.get("elements", []):
                        if parent_element.get("data", {}).get("id") == parent_id:
                            parent_data = parent_element.get("data", {})
                            hostname = parent_data.get("hostname")
                            if hostname and hostname.strip():
                                return hostname.strip()

                            # Traverse up to shelf level
                            grandparent_id = parent_data.get("parent")
                            if grandparent_id:
                                for grandparent_element in self.data.get("elements", []):
                                    if grandparent_element.get("data", {}).get("id") == grandparent_id:
                                        grandparent_data = grandparent_element.get("data", {})
                                        hostname = grandparent_data.get("hostname")
                                        if hostname and hostname.strip():
                                            return hostname.strip()
        return None

    def _get_node_type_from_port(self, port_id: str) -> str:
        """Get node_type from a port by traversing up to the shelf node"""
        # Find the port node in the cytoscape data
        for element in self.data.get("elements", []):
            if element.get("data", {}).get("id") == port_id:
                # Get the parent shelf node
                parent_id = element.get("data", {}).get("parent")
                if parent_id:
                    # Find the parent node
                    for parent_element in self.data.get("elements", []):
                        if parent_element.get("data", {}).get("id") == parent_id:
                            # Check if this is a shelf node
                            if parent_element.get("data", {}).get("type") == "shelf":
                                node_type = parent_element.get("data", {}).get("shelf_node_type", "N300_LB")
                                return node_type.upper()  # Ensure uppercase
                            # If not shelf, traverse up one more level
                            grandparent_id = parent_element.get("data", {}).get("parent")
                            if grandparent_id:
                                for grandparent_element in self.data.get("elements", []):
                                    if grandparent_element.get("data", {}).get("id") == grandparent_id:
                                        if grandparent_element.get("data", {}).get("type") == "shelf":
                                            node_type = grandparent_element.get("data", {}).get(
                                                "shelf_node_type", "N300_LB"
                                            )
                                            return node_type.upper()  # Ensure uppercase
        return "N300_LB"  # Default fallback


class DeploymentDataParser:
    """Parse Cytoscape.js data and extract deployment information"""

    def __init__(self, data: Dict):
        self.data = data
        self.nodes = {}
        self._parse_data()

    def _parse_data(self):
        """Parse Cytoscape data into nodes"""
        elements = self.data.get("elements", [])

        for element in elements:
            if "source" not in element.get("data", {}):
                # This is a node (not an edge)
                node_data = element.get("data", {})
                node_id = node_data.get("id")
                if node_id:
                    self.nodes[node_id] = element

    def _extract_host_info(self, node_id: str, node_data: Dict) -> Optional[Dict]:
        """Extract host information from a shelf node"""
        # Check if this is a shelf node
        if node_data.get("type") != "shelf":
            return None

        # Extract hostname and location information
        hostname = node_data.get("hostname")
        hall = node_data.get("hall")
        aisle = node_data.get("aisle")
        rack_num = node_data.get("rack_num") or node_data.get("rack")
        shelf_u = node_data.get("shelf_u")
        node_type = node_data.get("shelf_node_type")

        # Convert node_type to uppercase for internal storage
        if node_type:
            node_type = node_type.upper()

        # Normalize shelf_u to integer (strip 'U' prefix if present)
        if shelf_u is not None:
            if isinstance(shelf_u, str) and shelf_u.startswith("U"):
                shelf_u = int(shelf_u[1:])
            else:
                shelf_u = int(shelf_u)

        # Build host info dictionary with all available data
        host_info = {}

        # Add hostname if available (20-column format or 8-column format)
        if hostname and hostname.strip():
            host_info["hostname"] = hostname.strip()

        # Add location information if available (20-column format with full hierarchy)
        has_location = (
            hall and hall.strip() and aisle and aisle.strip() and rack_num is not None and shelf_u is not None
        )

        if has_location:
            host_info["hall"] = hall.strip()
            host_info["aisle"] = aisle.strip()
            host_info["rack_num"] = int(rack_num)
            host_info["shelf_u"] = shelf_u

        # Add node type if available
        if node_type:
            host_info["node_type"] = node_type

        # Return None if we have neither hostname nor location info
        if not host_info.get("hostname") and not has_location:
            return None

        return host_info

    def extract_hosts(self) -> List[Dict]:
        """Extract host information from shelf nodes"""
        hosts = []

        for node_id, node_element in self.nodes.items():
            node_data = node_element.get("data", {})
            host_info = self._extract_host_info(node_id, node_data)

            if host_info:
                hosts.append(host_info)

        return hosts


def export_cabling_descriptor_for_visualizer(cytoscape_data: Dict, filename_prefix: str = "cabling_descriptor") -> str:
    """Export CablingDescriptor from Cytoscape data"""
    if cluster_config_pb2 is None:
        raise ImportError("cluster_config_pb2 not available")

    parser = VisualizerCytoscapeDataParser(cytoscape_data)
    connections = parser.extract_connections()

    # Create ClusterDescriptor with full structure
    cluster_desc = cluster_config_pb2.ClusterDescriptor()

    # Create graph template
    template_name = "extracted_topology"
    graph_template = cluster_config_pb2.GraphTemplate()

    # Get unique hosts and their node types from connections
    host_info = {}
    for connection in connections:
        source_host = connection["source"]["hostname"]
        target_host = connection["target"]["hostname"]

        # Get node types from the connection data
        if source_host not in host_info:
            host_info[source_host] = connection["source"].get("node_type", "N300_LB")
        if target_host not in host_info:
            host_info[target_host] = connection["target"].get("node_type", "N300_LB")

    # Add child instances (one per host)
    host_name_map = {}  # Map actual hostname to host name (host_0, host_1, etc.)
    for i, (host, node_type) in enumerate(sorted(host_info.items())):
        host_name = f"host_{i}"
        host_name_map[host] = host_name

        child = graph_template.children.add()
        child.name = host_name
        child.node_ref.node_descriptor = node_type.upper()  # Ensure node_descriptor is capitalized

    # Add connections to graph template
    port_connections = graph_template.internal_connections["QSFP_DD"]  # Default port type
    for connection in connections:
        conn = port_connections.connections.add()

        # Source port - use mapped host name
        source_host_name = host_name_map[connection["source"]["hostname"]]
        conn.port_a.path.append(source_host_name)
        conn.port_a.tray_id = connection["source"]["tray_id"]
        conn.port_a.port_id = connection["source"]["port_id"]

        # Target port - use mapped host name
        target_host_name = host_name_map[connection["target"]["hostname"]]
        conn.port_b.path.append(target_host_name)
        conn.port_b.tray_id = connection["target"]["tray_id"]
        conn.port_b.port_id = connection["target"]["port_id"]

    # Add graph template to cluster descriptor
    cluster_desc.graph_templates[template_name].CopyFrom(graph_template)

    # Create root instance
    root_instance = cluster_config_pb2.GraphInstance()
    root_instance.template_name = template_name

    # Map each child to its host_id
    for i, (host, node_type) in enumerate(sorted(host_info.items())):
        child_mapping = cluster_config_pb2.ChildMapping()
        child_mapping.host_id = i
        root_instance.child_mappings[f"host_{i}"].CopyFrom(child_mapping)

    cluster_desc.root_instance.CopyFrom(root_instance)

    # Return the content directly
    return text_format.MessageToString(cluster_desc)


def export_deployment_descriptor_for_visualizer(
    cytoscape_data: Dict, filename_prefix: str = "deployment_descriptor"
) -> str:
    """Export DeploymentDescriptor from Cytoscape data

    Supports both 8-column format (hostname only) and 20-column format (hostname + location)
    """
    if deployment_pb2 is None:
        raise ImportError("deployment_pb2 not available")

    parser = DeploymentDataParser(cytoscape_data)
    hosts = parser.extract_hosts()

    # Create DeploymentDescriptor
    deployment_descriptor = deployment_pb2.DeploymentDescriptor()

    for host in hosts:
        host_proto = deployment_descriptor.hosts.add()

        # Set hostname if available (8-column or 20-column format)
        if "hostname" in host and host["hostname"]:
            host_proto.host = host["hostname"]

        # Set location information if available (20-column format)
        # These fields can coexist with hostname in the protobuf
        if "hall" in host and host["hall"]:
            host_proto.hall = host["hall"]
        if "aisle" in host and host["aisle"]:
            host_proto.aisle = host["aisle"]
        if "rack_num" in host:
            host_proto.rack = host["rack_num"]  # Use 'rack' field in protobuf
        if "shelf_u" in host:
            host_proto.shelf_u = host["shelf_u"]

        # Set node type if available
        if host.get("node_type"):
            host_proto.node_type = host["node_type"]

    # Return the content directly instead of a file path
    return text_format.MessageToString(deployment_descriptor)
