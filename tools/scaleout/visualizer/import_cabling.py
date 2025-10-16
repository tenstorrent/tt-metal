#!/usr/bin/env python3
"""
Network Cabling Visualizer - Cytoscape.js Implementation with Templates
Generates professional interactive network topology diagrams using cytoscape.js

Features:
- Template-based element positioning to reduce redundancy
- Hierarchical compound nodes (Racks > Shelf Units > Trays > Ports)
- Intelligent edge routing with automatic collision avoidance
- Interactive web interface with zoom, pan, and selection
- Color coding by cable length with visual hierarchy
"""

import csv
import argparse
import sys
import json
from collections import defaultdict
import os


class NetworkCablingCytoscapeVisualizer:
    """Professional network cabling topology visualizer using cytoscape.js with templates"""

    # Common dimensions used by all node types
    DEFAULT_SHELF_DIMENSIONS = {
        "width": "auto",  # Will be calculated based on tray layout
        "height": "auto",  # Will be calculated based on tray layout
        "padding": 15,  # Padding around trays inside shelf
    }

    # Common port dimensions (only spacing varies)
    DEFAULT_PORT_DIMENSIONS = {"width": 35, "height": 25}

    # Common tray dimensions for auto-calculated layouts
    DEFAULT_AUTO_TRAY_DIMENSIONS = {
        "width": "auto",  # Will be calculated based on port layout
        "height": "auto",  # Will be calculated based on port layout
        "spacing": 25,
        "padding": 8,  # Padding around ports inside tray
    }

    # Utility methods for common CSV parsing patterns
    @staticmethod
    def read_csv_lines(csv_file):
        """Read CSV file and return lines, skipping first two header lines"""
        with open(csv_file, "r") as file:
            lines = file.readlines()

        if len(lines) < 3:
            raise ValueError("CSV file must have at least 2 header lines and data")

        return lines

    @staticmethod
    def normalize_shelf_u(shelf_u_value):
        """Normalize shelf U value to numeric format (without U prefix)"""
        if not shelf_u_value:
            return "01"
        # Remove U prefix if present and ensure 2-digit format
        if shelf_u_value.startswith("U"):
            return shelf_u_value[1:].zfill(2)
        return shelf_u_value.zfill(2)

    @staticmethod
    def normalize_rack(rack_value):
        """Normalize rack number to 2-digit format"""
        if not rack_value:
            return "01"
        return rack_value.zfill(2)

    @staticmethod
    def safe_int(value, default=1):
        """Safely convert string to int with default fallback"""
        if value and value.isdigit():
            return int(value)
        return default

    @staticmethod
    def normalize_node_type(node_type, default="WH_GALAXY"):
        """Normalize node type to lowercase"""
        if not node_type:
            return default.lower()
        return node_type.lower()

    @staticmethod
    def create_connection_object(source_data, dest_data, cable_length="Unknown", cable_type="400G_AEC"):
        """Create standardized connection object"""
        return {"source": source_data, "destination": dest_data, "cable_length": cable_length, "cable_type": cable_type}

    def __init__(self, shelf_unit_type=None):
        # Data storage
        self.connections = []
        self.rack_units = {}  # rack_num -> set of shelf_u values
        self.shelf_units = {}  # hostname -> node_type for 8-column format
        self.shelf_unit_type = shelf_unit_type.lower() if shelf_unit_type else None
        self.csv_format = None  # Will be detected: '20_column' or '8_column'
        self.mixed_node_types = {}  # For 20-column format with mixed types
        self.dynamic_configs = {}  # For unknown node types discovered from CSV data

        # Cytoscape elements
        self.nodes = []
        self.edges = []

        # Define templates for different shelf unit types
        self.shelf_unit_configs = {
            "wh_galaxy": {
                "tray_count": 4,
                "port_count": 6,
                "tray_layout": "vertical",  # T1-T4 arranged vertically (top to bottom)
                # port_layout auto-inferred as 'horizontal' from vertical tray_layout
                "shelf_dimensions": self.DEFAULT_SHELF_DIMENSIONS.copy(),
                "tray_dimensions": {"width": 320, "height": 60, "spacing": 10},
                "port_dimensions": {**self.DEFAULT_PORT_DIMENSIONS, "spacing": 5},
            },
            "n300_lb": {
                "tray_count": 4,
                "port_count": 2,
                "tray_layout": "horizontal",  # T1-T4 arranged horizontally (left to right)
                # port_layout auto-inferred as 'vertical' from horizontal tray_layout
                "shelf_dimensions": self.DEFAULT_SHELF_DIMENSIONS.copy(),
                "tray_dimensions": self.DEFAULT_AUTO_TRAY_DIMENSIONS.copy(),
                "port_dimensions": {**self.DEFAULT_PORT_DIMENSIONS, "spacing": 15},
            },
            "n300_qb": {
                "tray_count": 4,
                "port_count": 2,
                "tray_layout": "horizontal",  # T1-T4 arranged horizontally (left to right)
                # port_layout auto-inferred as 'vertical' from horizontal tray_layout
                "shelf_dimensions": self.DEFAULT_SHELF_DIMENSIONS.copy(),
                "tray_dimensions": self.DEFAULT_AUTO_TRAY_DIMENSIONS.copy(),
                "port_dimensions": {**self.DEFAULT_PORT_DIMENSIONS, "spacing": 15},
            },
            "p150_qb": {
                "tray_count": 4,
                "port_count": 4,
                "tray_layout": "vertical",  # T1-T4 arranged vertically (T1 at bottom, T4 at top)
                # port_layout auto-inferred as 'horizontal' from vertical tray_layout
                "shelf_dimensions": self.DEFAULT_SHELF_DIMENSIONS.copy(),
                "tray_dimensions": self.DEFAULT_AUTO_TRAY_DIMENSIONS.copy(),
                "port_dimensions": {**self.DEFAULT_PORT_DIMENSIONS, "spacing": 15},
            },
            "p150_qb_global": {
                "tray_count": 4,
                "port_count": 4,
                "tray_layout": "horizontal",  # T1-T4 arranged horizontally (left to right)
                # port_layout auto-inferred as 'vertical' from horizontal tray_layout
                "shelf_dimensions": self.DEFAULT_SHELF_DIMENSIONS.copy(),
                "tray_dimensions": self.DEFAULT_AUTO_TRAY_DIMENSIONS.copy(),
                "port_dimensions": {**self.DEFAULT_PORT_DIMENSIONS, "spacing": 15},
            },
            "p150_qb_america": {
                "tray_count": 4,
                "port_count": 4,
                "tray_layout": "horizontal",  # T1-T4 arranged horizontally (left to right)
                # port_layout auto-inferred as 'vertical' from horizontal tray_layout
                "shelf_dimensions": self.DEFAULT_SHELF_DIMENSIONS.copy(),
                "tray_dimensions": self.DEFAULT_AUTO_TRAY_DIMENSIONS.copy(),
                "port_dimensions": {**self.DEFAULT_PORT_DIMENSIONS, "spacing": 15},
            },
            "bh_galaxy": {
                "tray_count": 4,
                "port_count": 14,
                "tray_layout": "vertical",  # T1-T4 arranged vertically (top to bottom)
                # port_layout auto-inferred as 'horizontal' from vertical tray_layout
                "shelf_dimensions": self.DEFAULT_SHELF_DIMENSIONS.copy(),
                "tray_dimensions": {"width": 320, "height": 60, "spacing": 10},
                "port_dimensions": {**self.DEFAULT_PORT_DIMENSIONS, "spacing": 5},
            },
        }

        # Get current shelf unit configuration - will be set after CSV parsing
        self.current_config = None

        # Calculate auto dimensions for trays based on port layout - will be done after config is set
        # This will be called in set_shelf_unit_type()

        # Element type templates - will be initialized after shelf unit type is determined
        self.element_templates = {}

        # Visual styling - colors for connection types
        self.intra_node_color = "#4CAF50"  # Green for connections within the same node
        self.inter_node_color = "#2196F3"  # Blue for connections between different nodes

        # Location information for nodes (hall, aisle info)
        self.node_locations = {}  # Will store hall/aisle info keyed by shelf_key

    def set_shelf_unit_type(self, shelf_unit_type):
        """Set the shelf unit type and initialize templates"""
        self.shelf_unit_type = shelf_unit_type.lower()

        # Get current shelf unit configuration
        self.current_config = self.shelf_unit_configs.get(self.shelf_unit_type, self.shelf_unit_configs["wh_galaxy"])

        # Calculate auto dimensions for trays based on port layout
        self.current_config = self.calculate_auto_dimensions(self.current_config)

        # Initialize element type templates
        if self.csv_format == "20_column":
            # Full hierarchy with racks
            self.element_templates = {
                "rack": {
                    "dimensions": {"width": 450, "height": 500, "spacing": 150},  # Generous spacing to prevent overlap
                    "position_type": "horizontal_sequence",  # Racks arranged left-to-right
                    "child_type": "shelf",
                    "style_class": "rack",
                },
                "shelf": {
                    "dimensions": self.current_config["shelf_dimensions"],
                    "dimensions_spacing": 60,  # Generous spacing to prevent overlap (shelves can be ~300px tall)
                    "position_type": "vertical_sequence",  # Shelves sorted descending, so higher U naturally goes to top
                    "child_type": "tray",
                    "style_class": f"shelf shelf-{self.shelf_unit_type}",
                },
                "tray": {
                    "dimensions": self.current_config["tray_dimensions"],
                    "position_type": "vertical_sequence"
                    if self.current_config["tray_layout"] == "vertical"
                    else "horizontal_sequence",
                    "child_type": "port",
                    "style_class": f"tray tray-{self.shelf_unit_type}",
                },
                "port": {
                    "dimensions": self.current_config["port_dimensions"],
                    "position_type": "vertical_sequence"
                    if self.infer_port_layout(self.current_config["tray_layout"]) == "vertical"
                    else "horizontal_sequence",
                    "child_type": None,
                    "style_class": "port",  # Unified port styling regardless of shelf type
                },
            }
            # Add spacing to shelf template
            self.element_templates["shelf"]["dimensions"]["spacing"] = self.element_templates["shelf"][
                "dimensions_spacing"
            ]
        else:
            # Shelf-only format for 8-column
            self.element_templates = {
                "shelf": {
                    "dimensions": self.current_config["shelf_dimensions"],
                    "dimensions_spacing": 50,  # More spacing between independent shelf units
                    "position_type": "horizontal_sequence",  # Shelf units arranged left-to-right
                    "child_type": "tray",
                    "style_class": f"shelf shelf-{self.shelf_unit_type}",
                },
                "tray": {
                    "dimensions": self.current_config["tray_dimensions"],
                    "position_type": "vertical_sequence"
                    if self.current_config["tray_layout"] == "vertical"
                    else "horizontal_sequence",
                    "child_type": "port",
                    "style_class": f"tray tray-{self.shelf_unit_type}",
                },
                "port": {
                    "dimensions": self.current_config["port_dimensions"],
                    "position_type": "vertical_sequence"
                    if self.infer_port_layout(self.current_config["tray_layout"]) == "vertical"
                    else "horizontal_sequence",
                    "child_type": None,
                    "style_class": "port",  # Unified port styling regardless of shelf type
                },
            }
            # Add spacing to shelf template
            self.element_templates["shelf"]["dimensions"]["spacing"] = self.element_templates["shelf"][
                "dimensions_spacing"
            ]

    def detect_csv_format(self, csv_file):
        """Detect CSV format by examining the first data row"""
        try:
            with open(csv_file, "r") as file:
                lines = file.readlines()

                # Skip headers and find first data row
                for line in lines[2:]:  # Skip first two header lines
                    line = line.strip()
                    if not line:
                        continue

                    # Count columns by splitting
                    columns = line.split(",")
                    column_count = len([col for col in columns if col])  # Count non-empty columns

                    if column_count >= 20:
                        return "20_column"
                    elif column_count >= 8:
                        return "8_column"
                    else:
                        raise ValueError(f"Unrecognized CSV format: {column_count} columns found")

            raise ValueError("No data rows found in CSV")

        except Exception as e:
            return None

    def analyze_and_create_dynamic_config(self, node_type, connections):
        """Analyze CSV data to create dynamic configuration for unknown node types"""

        # Track maximum tray and port numbers seen for this node type
        max_tray = 0
        max_port = 0

        for connection in connections:
            # Check both source and destination
            if connection["source"].get("node_type") == node_type:
                max_tray = max(max_tray, connection["source"]["tray"])
                max_port = max(max_port, connection["source"]["port"])
            if connection["destination"].get("node_type") == node_type:
                max_tray = max(max_tray, connection["destination"]["tray"])
                max_port = max(max_port, connection["destination"]["port"])

        # Determine layout based on tray/port ratios (heuristic)
        # If more trays than ports, likely horizontal tray layout
        # If more ports than trays, likely vertical tray layout
        if max_tray >= max_port:
            tray_layout = "horizontal"
        else:
            tray_layout = "vertical"

        # Port layout is automatically inferred from tray layout
        port_layout = self.infer_port_layout(tray_layout)

        # Create dynamic configuration
        dynamic_config = {
            "tray_count": max_tray,
            "port_count": max_port,
            "tray_layout": tray_layout,
            "port_layout": port_layout,
            "shelf_dimensions": self.DEFAULT_SHELF_DIMENSIONS.copy(),
            "tray_dimensions": self.DEFAULT_AUTO_TRAY_DIMENSIONS.copy(),
            "port_dimensions": {**self.DEFAULT_PORT_DIMENSIONS, "spacing": 15},
        }

        # Store the dynamic configuration
        self.dynamic_configs[node_type] = dynamic_config
        self.shelf_unit_configs[node_type] = dynamic_config  # Also add to main configs

        return dynamic_config

    def get_unknown_node_types(self):
        """Return list of node types that were dynamically created (unknown)"""
        return list(self.dynamic_configs.keys())

    def infer_port_layout(self, tray_layout):
        """Automatically infer port layout from tray layout"""
        # If trays are vertical, ports should be horizontal
        # If trays are horizontal, ports should be vertical
        return "horizontal" if tray_layout == "vertical" else "vertical"

    def calculate_auto_dimensions(self, config):
        """Calculate automatic tray and shelf dimensions based on layout"""
        # Create a copy of config for modifications
        config = config.copy()

        # Step 1: Calculate tray dimensions if needed
        if config["tray_dimensions"].get("width") == "auto" or config["tray_dimensions"].get("height") == "auto":
            port_width = config["port_dimensions"]["width"]
            port_height = config["port_dimensions"]["height"]
            port_spacing = config["port_dimensions"]["spacing"]
            port_count = config["port_count"]
            tray_padding = config["tray_dimensions"].get("padding", 8)

            # Infer port layout from tray layout
            port_layout = self.infer_port_layout(config["tray_layout"])

            # Calculate based on port layout
            if port_layout == "vertical":
                # Ports stacked vertically: width = port_width + padding, height = ports + spacing + padding
                auto_tray_width = port_width + 2 * tray_padding
                auto_tray_height = port_count * port_height + (port_count - 1) * port_spacing + 2 * tray_padding
            else:  # horizontal
                # Ports arranged horizontally: width = ports + spacing + padding, height = port_height + padding
                auto_tray_width = port_count * port_width + (port_count - 1) * port_spacing + 2 * tray_padding
                auto_tray_height = port_height + 2 * tray_padding

            config["tray_dimensions"] = config["tray_dimensions"].copy()

            if config["tray_dimensions"].get("width") == "auto":
                config["tray_dimensions"]["width"] = auto_tray_width
            if config["tray_dimensions"].get("height") == "auto":
                config["tray_dimensions"]["height"] = auto_tray_height

        # Step 2: Calculate shelf dimensions based on tray layout
        if config["shelf_dimensions"].get("width") == "auto" or config["shelf_dimensions"].get("height") == "auto":
            tray_width = config["tray_dimensions"]["width"]
            tray_height = config["tray_dimensions"]["height"]
            tray_spacing = config["tray_dimensions"]["spacing"]
            tray_count = config["tray_count"]
            shelf_padding = config["shelf_dimensions"].get("padding", 15)

            # Calculate based on tray layout
            if config["tray_layout"] == "vertical":
                # Trays stacked vertically: width = tray_width + padding, height = trays + spacing + padding
                auto_shelf_width = tray_width + 2 * shelf_padding
                auto_shelf_height = tray_count * tray_height + (tray_count - 1) * tray_spacing + 2 * shelf_padding
            else:  # horizontal
                # Trays arranged horizontally: width = trays + spacing + padding, height = tray_height + padding
                auto_shelf_width = tray_count * tray_width + (tray_count - 1) * tray_spacing + 2 * shelf_padding
                auto_shelf_height = tray_height + 2 * shelf_padding

            config["shelf_dimensions"] = config["shelf_dimensions"].copy()

            if config["shelf_dimensions"].get("width") == "auto":
                config["shelf_dimensions"]["width"] = auto_shelf_width
            if config["shelf_dimensions"].get("height") == "auto":
                config["shelf_dimensions"]["height"] = auto_shelf_height

        return config

    def parse_csv(self, csv_file):
        """Parse CSV file containing cabling connections with format auto-detection"""
        try:
            # First, detect the CSV format
            self.csv_format = self.detect_csv_format(csv_file)
            if not self.csv_format:
                return []

            print(f"Detected CSV format: {self.csv_format}")

            if self.csv_format == "20_column":
                return self.parse_20_column_csv(csv_file)
            elif self.csv_format == "8_column":
                return self.parse_8_column_csv(csv_file)
            else:
                raise ValueError(f"Unsupported CSV format: {self.csv_format}")

        except Exception as e:
            print(f"Error parsing CSV file: {e}")
            return []

    def parse_20_column_csv(self, csv_file):
        """Parse 20-column CSV with full hierarchy including hostname (Hostname,Hall,Aisle,Rack,Shelf U,Tray,Port,Label,Node Type,...)"""
        try:
            lines = self.read_csv_lines(csv_file)
            node_types_seen = set()

            # Process data lines starting from line 3 (index 2)
            for line in lines[2:]:
                line = line.strip()
                if not line:
                    continue

                # Split by comma to get all columns
                row_values = line.split(",")

                # CSV structure: Hostname,Hall,Aisle,Rack,Shelf U,Tray,Port,Label,Node Type,Hostname,Hall,Aisle,Rack,Shelf U,Tray,Port,Label,Node Type,Cable Length,Cable Type
                if len(row_values) >= 18:
                    # Source columns: positions 0,1,2,3,4,5,6,7,8 = Hostname,Hall,Aisle,Rack,Shelf U,Tray,Port,Label,Node Type
                    src_hostname = row_values[0] if row_values[0] else ""
                    src_hall = row_values[1] if row_values[1] else ""
                    src_aisle = row_values[2] if row_values[2] else ""
                    src_rack = self.normalize_rack(row_values[3])
                    src_shelf_u = self.normalize_shelf_u(row_values[4])
                    src_tray = self.safe_int(row_values[5])
                    src_port = self.safe_int(row_values[6])
                    src_label = row_values[7] if row_values[7] else f"{src_rack}{src_shelf_u}-{src_tray}-{src_port}"
                    src_node_type = self.normalize_node_type(row_values[8])

                    # Destination columns: positions 9,10,11,12,13,14,15,16,17 = Hostname,Hall,Aisle,Rack,Shelf U,Tray,Port,Label,Node Type
                    dst_hostname = row_values[9] if len(row_values) > 9 and row_values[9] else ""
                    dst_hall = row_values[10] if len(row_values) > 10 and row_values[10] else ""
                    dst_aisle = row_values[11] if len(row_values) > 11 and row_values[11] else ""
                    dst_rack = self.normalize_rack(row_values[12] if len(row_values) > 12 else "")
                    dst_shelf_u = self.normalize_shelf_u(row_values[13] if len(row_values) > 13 else "")
                    dst_tray = self.safe_int(row_values[14] if len(row_values) > 14 else "")
                    dst_port = self.safe_int(row_values[15] if len(row_values) > 15 else "")
                    dst_label = (
                        row_values[16]
                        if len(row_values) > 16 and row_values[16]
                        else f"{dst_rack}{dst_shelf_u}-{dst_tray}-{dst_port}"
                    )
                    dst_node_type = self.normalize_node_type(row_values[17] if len(row_values) > 17 else "")

                    # Cable info: positions 18,19 = Cable Length,Cable Type
                    cable_length = row_values[18] if len(row_values) > 18 and row_values[18] else "Unknown"
                    cable_type = row_values[19] if len(row_values) > 19 and row_values[19] else "400G_AEC"

                    # Track node types for each shelf unit
                    src_shelf_key = f"{src_rack}_{src_shelf_u}"
                    dst_shelf_key = f"{dst_rack}_{dst_shelf_u}"
                    self.mixed_node_types[src_shelf_key] = src_node_type
                    self.mixed_node_types[dst_shelf_key] = dst_node_type

                    # Track location information for each shelf unit
                    self.node_locations[src_shelf_key] = {
                        "hostname": src_hostname,
                        "hall": src_hall,
                        "aisle": src_aisle,
                        "rack_num": src_rack,
                        "shelf_u": src_shelf_u,
                    }
                    self.node_locations[dst_shelf_key] = {
                        "hostname": dst_hostname,
                        "hall": dst_hall,
                        "aisle": dst_aisle,
                        "rack_num": dst_rack,
                        "shelf_u": dst_shelf_u,
                    }

                    node_types_seen.add(src_node_type)
                    node_types_seen.add(dst_node_type)

                    connection = {
                        "source": {
                            "hostname": src_hostname,
                            "hall": src_hall,
                            "aisle": src_aisle,
                            "rack_num": src_rack,
                            "shelf_u": src_shelf_u,
                            "tray": src_tray,
                            "port": src_port,
                            "label": src_label,
                            "node_type": src_node_type,
                        },
                        "destination": {
                            "hostname": dst_hostname,
                            "hall": dst_hall,
                            "aisle": dst_aisle,
                            "rack_num": dst_rack,
                            "shelf_u": dst_shelf_u,
                            "tray": dst_tray,
                            "port": dst_port,
                            "label": dst_label,
                            "node_type": dst_node_type,
                        },
                        "cable_length": cable_length,
                        "cable_type": cable_type,
                    }

                    self.connections.append(connection)

                    # Track rack units for layout
                    self.rack_units.setdefault(src_rack, set()).add(src_shelf_u)
                    self.rack_units.setdefault(dst_rack, set()).add(dst_shelf_u)

                # Create dynamic configurations for unknown node types
                for node_type in node_types_seen:
                    if node_type not in self.shelf_unit_configs:
                        self.analyze_and_create_dynamic_config(node_type, self.connections)

                # Set a default shelf unit type if not specified (use the most common one)
                if not self.shelf_unit_type and node_types_seen:
                    self.shelf_unit_type = list(node_types_seen)[0]  # Use first one found
                elif not self.shelf_unit_type:
                    self.shelf_unit_type = "wh_galaxy"

                # Initialize templates with the default type
                self.set_shelf_unit_type(self.shelf_unit_type)

            return self.connections

        except Exception as e:
            return []

    def parse_8_column_csv(self, csv_file):
        """Parse 8-column CSV with hostname format (Hostname,Tray,Port,Node Type,...)"""
        try:
            lines = self.read_csv_lines(csv_file)
            node_types_seen = set()

            # Process data lines starting from line 3 (index 2)
            for line in lines[2:]:
                line = line.strip()
                if not line:
                    continue

                # Split by comma to get all columns
                row_values = line.split(",")

                # CSV structure: Hostname,Tray,Port,Node Type,Hostname,Tray,Port,Node Type
                if len(row_values) >= 8:
                    # Source columns: positions 0,1,2,3 = Hostname,Tray,Port,Node Type
                    src_hostname = row_values[0] if row_values[0] else "shelf-01"
                    src_tray = self.safe_int(row_values[1])
                    src_port = self.safe_int(row_values[2])
                    src_node_type = self.normalize_node_type(row_values[3])

                    # Destination columns: positions 4,5,6,7 = Hostname,Tray,Port,Node Type
                    dst_hostname = row_values[4] if len(row_values) > 4 and row_values[4] else "shelf-02"
                    dst_tray = self.safe_int(row_values[5] if len(row_values) > 5 else "")
                    dst_port = self.safe_int(row_values[6] if len(row_values) > 6 else "")
                    dst_node_type = self.normalize_node_type(row_values[7] if len(row_values) > 7 else "")

                    # Track node types for each shelf unit (hostname)
                    self.shelf_units[src_hostname] = src_node_type
                    self.shelf_units[dst_hostname] = dst_node_type

                    node_types_seen.add(src_node_type)
                    node_types_seen.add(dst_node_type)

                    # Create source and destination data objects
                    source_data = {
                        "hostname": src_hostname,
                        "tray": src_tray,
                        "port": src_port,
                        "label": f"{src_hostname}-{src_tray}-{src_port}",
                        "node_type": src_node_type,
                    }

                    dest_data = {
                        "hostname": dst_hostname,
                        "tray": dst_tray,
                        "port": dst_port,
                        "label": f"{dst_hostname}-{dst_tray}-{dst_port}",
                        "node_type": dst_node_type,
                    }

                    connection = self.create_connection_object(source_data, dest_data)

                    self.connections.append(connection)

                # Create dynamic configurations for unknown node types
                for node_type in node_types_seen:
                    if node_type not in self.shelf_unit_configs:
                        self.analyze_and_create_dynamic_config(node_type, self.connections)

                # Set a default shelf unit type if not specified (use the most common one)
                if not self.shelf_unit_type and node_types_seen:
                    self.shelf_unit_type = list(node_types_seen)[0]  # Use first one found
                elif not self.shelf_unit_type:
                    self.shelf_unit_type = "wh_galaxy"

                # Initialize templates with the default type
                self.set_shelf_unit_type(self.shelf_unit_type)

            return self.connections

        except Exception as e:
            return []

    def parse_connection_label(self, label):
        """Parse connection label formats: 120A03U14-2-3 or SC_Floor_5A01U32-2-1"""
        try:
            if label.startswith("120A"):
                # Format: 120A03U14-2-3
                parts = label[4:].split("-")  # Remove "120A" prefix

                # Extract rack and shelf from first part (e.g., "03U14")
                rack_shelf = parts[0]
                if "U" in rack_shelf:
                    rack_num, shelf_u = rack_shelf.split("U")
                    shelf_u = shelf_u.zfill(2)  # Ensure consistent formatting, numeric only
                else:
                    return None

                tray = int(parts[1]) if len(parts) > 1 else 1
                port = int(parts[2]) if len(parts) > 2 else 1

                return {
                    "rack": rack_num.zfill(2),  # Ensure consistent formatting
                    "shelf": shelf_u,
                    "tray": tray,
                    "port": port,
                }

            elif "A" in label and "U" in label and "-" in label:
                # Format: SC_Floor_5A01U32-2-1
                # Find the A and U positions to extract rack and shelf
                a_pos = label.rfind("A")  # Find last A
                if a_pos == -1:
                    return None

                # Extract everything after the last A
                after_a = label[a_pos + 1 :]  # e.g., "01U32-2-1"

                # Split by dash to get rack_shelf and tray/port parts
                parts = after_a.split("-")
                if len(parts) < 3:
                    return None

                rack_shelf_part = parts[0]  # e.g., "01U32"

                if "U" in rack_shelf_part:
                    rack_num, shelf_u = rack_shelf_part.split("U")
                    shelf_u = shelf_u.zfill(2)  # Ensure consistent formatting, numeric only
                else:
                    return None

                tray = int(parts[1]) if len(parts) > 1 else 1
                port = int(parts[2]) if len(parts) > 2 else 1

                return {
                    "rack": rack_num.zfill(2),  # Ensure consistent formatting
                    "shelf": shelf_u,
                    "tray": tray,
                    "port": port,
                }
            else:
                return None

        except (IndexError, ValueError) as e:
            return None

    def generate_node_id(self, node_type, *args):
        """Generate consistent node IDs for cytoscape elements"""
        if node_type == "port" and len(args) >= 3:
            # Format: <label>-tray#-port#
            return f"{args[0]}-tray{args[1]}-port{args[2]}"
        elif node_type == "tray" and len(args) >= 2:
            # Format: <label>-tray#
            return f"{args[0]}-tray{args[1]}"
        elif node_type == "shelf":
            # Format: <label> - for hierarchical format, use rack_U_shelf format (shelf already numeric)
            if len(args) >= 2:
                return f"{args[0]}_U{args[1]}"
            else:
                return str(args[0])
        else:
            # Fallback to original format for other cases
            return f"{node_type}_{'_'.join(str(arg) for arg in args)}"

    def calculate_position_in_sequence(self, element_type, index, parent_x=0, parent_y=0):
        """Calculate position for an element in a sequence based on its template"""
        template = self.element_templates[element_type]
        dimensions = template["dimensions"]
        position_type = template["position_type"]

        if position_type == "horizontal_sequence":
            # Elements arranged left-to-right (e.g., racks, ports)
            x = parent_x + index * (dimensions["width"] + dimensions["spacing"])
            y = parent_y

        elif position_type == "vertical_sequence":
            # Elements arranged top-to-bottom (e.g., trays)
            x = parent_x
            y = parent_y + index * (dimensions["height"] + dimensions["spacing"])

        elif position_type == "vertical_sequence_reversed":
            # Elements arranged bottom-to-top (e.g., shelves with lower U at bottom)
            x = parent_x
            # Note: This will be corrected in the calling function with total count
            y = parent_y + index * (dimensions["height"] + dimensions["spacing"])

        return x, y

    def get_child_positions_for_parent(self, parent_type, child_indices, parent_x, parent_y):
        """Get all child positions for a parent element using templates"""
        template = self.element_templates[parent_type]
        child_type = template["child_type"]

        if not child_type:
            return []

        child_positions = []
        for index, child_id in enumerate(child_indices):
            x, y = self.calculate_position_in_sequence(child_type, index, parent_x, parent_y)
            child_positions.append((child_id, x, y))

        # Handle reversed sequences (e.g., shelves)
        child_template = self.element_templates[child_type]
        if child_template["position_type"] == "vertical_sequence_reversed":
            # Reverse the Y positions so lower indices are at bottom
            total_count = len(child_indices)
            corrected_positions = []
            for child_id, x, original_y in child_positions:
                # Calculate position from bottom instead of top
                corrected_index = total_count - 1 - child_positions.index((child_id, x, original_y))
                _, corrected_y = self.calculate_position_in_sequence(child_type, corrected_index, parent_x, parent_y)
                corrected_positions.append((child_id, x, corrected_y))
            return corrected_positions

        return child_positions

    def create_node_from_template(self, node_type, node_id, parent_id, label, x, y, **extra_data):
        """Create a cytoscape node using element template"""
        template = self.element_templates[node_type]

        node_data = {"id": node_id, "label": label, "type": node_type, **extra_data}

        # Add parent relationship if specified
        if parent_id:
            node_data["parent"] = parent_id

        node = {
            "data": node_data,
            "classes": template["style_class"],
            "position": {"x": x + template["dimensions"]["width"] / 2, "y": y + template["dimensions"]["height"] / 2},
        }

        return node

    def create_hierarchical_nodes(self):
        """Create hierarchical compound nodes using templates for positioning"""

        if self.csv_format == "20_column":
            self.create_20_column_hierarchical_nodes()
        elif self.csv_format == "8_column":
            self.create_8_column_hierarchical_nodes()

    def create_20_column_hierarchical_nodes(self):
        """Create full hierarchy nodes for 20-column format (racks -> shelves -> trays -> ports)"""

        # Get sorted rack numbers for consistent ordering
        rack_numbers = sorted(self.rack_units.keys())

        # Calculate rack positions using template
        rack_positions = []
        for rack_idx, rack_num in enumerate(rack_numbers):
            rack_x, rack_y = self.calculate_position_in_sequence("rack", rack_idx)
            rack_positions.append((rack_num, rack_x, rack_y))

        # Create all nodes using template-based approach
        for rack_num, rack_x, rack_y in rack_positions:
            # Get shelf units for this rack to extract hall/aisle info
            # Sort in descending order so higher U numbers are at top
            shelf_units = sorted(self.rack_units[rack_num], reverse=True)

            # Get hall and aisle info from the first shelf in this rack (if available)
            hall = ""
            aisle = ""
            if shelf_units:
                first_shelf_key = f"{rack_num}_{shelf_units[0]}"
                first_shelf_info = self.node_locations.get(first_shelf_key, {})
                hall = first_shelf_info.get("hall", "")
                aisle = first_shelf_info.get("aisle", "")

            # Create rack node with location info
            rack_id = self.generate_node_id("rack", rack_num)
            rack_node = self.create_node_from_template(
                "rack", rack_id, None, f"Rack {rack_num}", rack_x, rack_y, rack_num=rack_num, hall=hall, aisle=aisle
            )
            self.nodes.append(rack_node)

            # Calculate shelf positions
            shelf_positions = self.get_child_positions_for_parent("rack", shelf_units, rack_x, rack_y)

            for shelf_u, shelf_x, shelf_y in shelf_positions:
                # Get the node type and location info for this specific shelf
                shelf_key = f"{rack_num}_{shelf_u}"
                shelf_node_type = self.mixed_node_types.get(shelf_key, self.shelf_unit_type)
                shelf_config = self.shelf_unit_configs.get(shelf_node_type, self.current_config)
                location_info = self.node_locations.get(shelf_key, {})
                hostname = location_info.get("hostname", "")

                # Create shelf node with hostname
                shelf_id = self.generate_node_id("shelf", rack_num, shelf_u)
                shelf_label = f"{hostname}" if hostname else f"Shelf {shelf_u}"
                shelf_node = self.create_node_from_template(
                    "shelf",
                    shelf_id,
                    rack_id,
                    shelf_label,
                    shelf_x,
                    shelf_y,
                    rack_num=rack_num,
                    shelf_u=shelf_u,
                    shelf_node_type=shelf_node_type,
                    hostname=hostname,
                    hall=location_info.get("hall", ""),
                    aisle=location_info.get("aisle", ""),
                )
                self.nodes.append(shelf_node)

                # Create trays based on this shelf's specific configuration
                tray_count = shelf_config["tray_count"]
                tray_ids = list(range(1, tray_count + 1))  # T1, T2, T3, T4 (or however many)
                tray_positions = self.get_child_positions_for_parent("shelf", tray_ids, shelf_x, shelf_y)

                for tray_id, tray_x, tray_y in tray_positions:
                    # Create tray node - use the actual shelf_id
                    tray_node_id = self.generate_node_id("tray", shelf_id, tray_id)
                    tray_node = self.create_node_from_template(
                        "tray",
                        tray_node_id,
                        shelf_id,
                        f"T{tray_id}",
                        tray_x,
                        tray_y,
                        rack_num=rack_num,
                        shelf_u=shelf_u,
                        tray=tray_id,
                        shelf_node_type=shelf_node_type,
                        hostname=hostname,
                    )
                    self.nodes.append(tray_node)

                    # Create ports based on this shelf's specific configuration
                    port_count = shelf_config["port_count"]
                    port_ids = list(range(1, port_count + 1))  # P1, P2, ... (based on config)
                    port_positions = self.get_child_positions_for_parent("tray", port_ids, tray_x, tray_y)

                    for port_id, port_x, port_y in port_positions:
                        # Create port node - use the actual shelf_id
                        port_node_id = self.generate_node_id("port", shelf_id, tray_id, port_id)
                        port_node = self.create_node_from_template(
                            "port",
                            port_node_id,
                            tray_node_id,
                            f"P{port_id}",
                            port_x,
                            port_y,
                            rack_num=rack_num,
                            shelf_u=shelf_u,
                            tray=tray_id,
                            port=port_id,
                            shelf_node_type=shelf_node_type,
                            hostname=hostname,
                        )
                        self.nodes.append(port_node)

    def create_8_column_hierarchical_nodes(self):
        """Create shelf-only hierarchy nodes for 8-column format (shelves -> trays -> ports)"""

        # Get sorted hostnames for consistent ordering
        hostnames = sorted(self.shelf_units.keys())

        # Calculate shelf positions using template
        shelf_positions = []
        for shelf_idx, hostname in enumerate(hostnames):
            shelf_x, shelf_y = self.calculate_position_in_sequence("shelf", shelf_idx)
            shelf_positions.append((hostname, shelf_x, shelf_y))

        # Create all nodes using template-based approach (no racks)
        for hostname, shelf_x, shelf_y in shelf_positions:
            # Get the node type for this specific shelf
            shelf_node_type = self.shelf_units.get(hostname, self.shelf_unit_type)
            shelf_config = self.shelf_unit_configs.get(shelf_node_type, self.current_config)

            # Create shelf node (no parent)
            shelf_id = self.generate_node_id("shelf", hostname)
            shelf_node = self.create_node_from_template(
                "shelf",
                shelf_id,
                None,
                f"{hostname}",
                shelf_x,
                shelf_y,
                hostname=hostname,
                shelf_node_type=shelf_node_type,
            )
            self.nodes.append(shelf_node)

            # Create trays based on this shelf's specific configuration
            tray_count = shelf_config["tray_count"]
            tray_ids = list(range(1, tray_count + 1))  # T1, T2, T3, T4 (or however many)
            tray_positions = self.get_child_positions_for_parent("shelf", tray_ids, shelf_x, shelf_y)

            for tray_id, tray_x, tray_y in tray_positions:
                # Create tray node
                tray_node_id = self.generate_node_id("tray", hostname, tray_id)
                tray_node = self.create_node_from_template(
                    "tray",
                    tray_node_id,
                    shelf_id,
                    f"T{tray_id}",
                    tray_x,
                    tray_y,
                    tray=tray_id,
                    shelf_node_type=shelf_node_type,
                )
                self.nodes.append(tray_node)

                # Create ports based on this shelf's specific configuration
                port_count = shelf_config["port_count"]
                port_ids = list(range(1, port_count + 1))  # P1, P2, ... (based on config)
                port_positions = self.get_child_positions_for_parent("tray", port_ids, tray_x, tray_y)

                for port_id, port_x, port_y in port_positions:
                    # Create port node
                    port_node_id = self.generate_node_id("port", hostname, tray_id, port_id)
                    port_node = self.create_node_from_template(
                        "port",
                        port_node_id,
                        tray_node_id,
                        f"P{port_id}",
                        port_x,
                        port_y,
                        tray=tray_id,
                        port=port_id,
                        shelf_node_type=shelf_node_type,
                    )
                    self.nodes.append(port_node)

    def create_connection_edges(self):
        """Create edges representing connections between ports"""

        for i, connection in enumerate(self.connections, 1):
            if self.csv_format == "20_column":
                # 20-column format: rack/shelf/tray/port hierarchy
                src_shelf_label = f"{connection['source']['rack_num']}_U{connection['source']['shelf_u']}"
                src_port_id = self.generate_node_id(
                    "port", src_shelf_label, connection["source"]["tray"], connection["source"]["port"]
                )
                dst_shelf_label = f"{connection['destination']['rack_num']}_U{connection['destination']['shelf_u']}"
                dst_port_id = self.generate_node_id(
                    "port", dst_shelf_label, connection["destination"]["tray"], connection["destination"]["port"]
                )
            elif self.csv_format == "8_column":
                # 8-column format: hostname/tray/port hierarchy (no racks)
                src_port_id = self.generate_node_id(
                    "port", connection["source"]["hostname"], connection["source"]["tray"], connection["source"]["port"]
                )
                dst_port_id = self.generate_node_id(
                    "port",
                    connection["destination"]["hostname"],
                    connection["destination"]["tray"],
                    connection["destination"]["port"],
                )

            # Determine connection color based on whether ports are on the same node
            # Handle different CSV formats
            if self.csv_format == "20_column":
                # 20-column format: compare rack_num and shelf_u
                source_node_id = f"{connection['source']['rack_num']}_{connection['source']['shelf_u']}"
                dest_node_id = f"{connection['destination']['rack_num']}_{connection['destination']['shelf_u']}"
            elif self.csv_format == "8_column":
                # 8-column format: compare hostnames
                source_node_id = connection["source"]["hostname"]
                dest_node_id = connection["destination"]["hostname"]
            else:
                # Fallback: assume different nodes
                source_node_id = "unknown_source"
                dest_node_id = "unknown_dest"

            if source_node_id == dest_node_id:
                color = self.intra_node_color  # Same node - green
            else:
                color = self.inter_node_color  # Different nodes - blue

            # Calculate alternating direction properties
            # Alternate between positive and negative control point distance
            direction_multiplier = 1

            # Base control point distance (can be adjusted)
            base_distance = 60
            control_point_distance = base_distance * direction_multiplier

            # Create alternating control point arrays for unbundled-bezier
            control_distances = [control_point_distance, -control_point_distance]
            control_weights = [0.25, 0.75]

            #

            # Generate random label position along the edge (0.2 to 0.8 to avoid endpoints)
            import random

            random.seed(i)  # Use connection index as seed for consistent positioning
            label_position = 0.2 + (random.random() * 0.6)  # Random value between 0.2 and 0.8

            # Create edge with comprehensive connection info
            edge_id = f"connection_{i}"
            edge_data = {
                "data": {
                    "id": edge_id,
                    "source": src_port_id,
                    "target": dst_port_id,
                    "label": f"#{i}",
                    "cable_length": connection["cable_length"],
                    "cable_type": connection["cable_type"],
                    "connection_number": i,
                    "color": color,
                    "source_info": connection["source"]["label"],  # Use CSV label
                    "destination_info": connection["destination"]["label"],  # Use CSV label
                    # Add hostname info if available (20-column format)
                    "source_hostname": connection["source"].get("hostname", ""),
                    "destination_hostname": connection["destination"].get("hostname", ""),
                    # Alternating direction properties
                    "control_point_distances": control_distances,
                    "control_point_weights": control_weights,
                    "direction_multiplier": direction_multiplier,
                    # Random label positioning
                    "label_position": label_position,
                },
                "classes": "connection",
            }

            self.edges.append(edge_data)

    def generate_cytoscape_data(self):
        """Generate complete cytoscape.js data structure"""
        self.nodes = []
        self.edges = []

        # Create hierarchical nodes using templates
        self.create_hierarchical_nodes()

        # Create connection edges
        self.create_connection_edges()

        return {"elements": self.nodes + self.edges}

    def generate_visualization_data(self):
        """Generate cytoscape.js visualization data structure (library method)"""

        # Generate cytoscape data
        cytoscape_data = self.generate_cytoscape_data()

        # Add metadata about the shelf unit type and configuration
        cytoscape_data["metadata"] = {
            "csv_format": self.csv_format,
            "shelf_unit_type": self.shelf_unit_type.title() if self.shelf_unit_type else "Auto-detected",
            "configuration": self.current_config,
            "generation_info": {
                "nodes": len(self.nodes),
                "edges": len(self.edges),
                "template_types": list(self.element_templates.keys()),
            },
        }

        # Add mixed node types info for 20-column format
        if self.csv_format == "20_column" and self.mixed_node_types:
            cytoscape_data["metadata"]["mixed_node_types"] = self.mixed_node_types
        elif self.csv_format == "8_column" and self.shelf_units:
            cytoscape_data["metadata"]["shelf_node_types"] = self.shelf_units

        return cytoscape_data

    def create_diagram(self, output_file="templated_demo_data.json"):
        """Create network cabling topology diagram using cytoscape.js with templates"""

        # Generate cytoscape data
        cytoscape_data = self.generate_visualization_data()

        # For demonstration, save the data structure
        with open(output_file, "w") as f:
            json.dump(cytoscape_data, f, indent=2)

        return cytoscape_data


def main():
    """Main entry point with command line interface for template demo"""
    parser = argparse.ArgumentParser(
        description="Demonstrate template-based generation of Cytoscape.js elements with auto-detected CSV format and node types"
    )
    parser.add_argument(
        "csv_file", help="Input CSV cabling file (supports 20-column with hostname or 8-column hostname format)"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="templated_demo_data.json",
        help="Output JSON file for generated Cytoscape.js data (default: templated_demo_data.json)",
    )

    args = parser.parse_args()

    # Create visualizer without specifying shelf unit type (will be auto-detected from CSV)
    visualizer = NetworkCablingCytoscapeVisualizer()

    connections = visualizer.parse_csv(args.csv_file)
    if not connections:
        sys.exit(1)

    # Generate hierarchical nodes using the template system
    visualizer.create_diagram(args.output)


if __name__ == "__main__":
    main()
