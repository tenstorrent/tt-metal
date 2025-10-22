#!/usr/bin/env python3
"""
Flask web server for Network Cabling Visualizer
Provides CSV upload interface and generates JSON visualization data on-the-fly
"""

import os
import sys
import tempfile
import argparse
from flask import Flask, request, jsonify, render_template, send_from_directory, Response
import traceback

# Add the parent directory to sys.path to import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our existing templating system
from import_cabling import NetworkCablingCytoscapeVisualizer

# Import export functionality
try:
    # Test if export functionality is available
    from export_descriptors import export_cabling_descriptor_for_visualizer, export_deployment_descriptor_for_visualizer

    EXPORT_AVAILABLE = True
except ImportError as e:
    EXPORT_AVAILABLE = False

app = Flask(__name__)
# No CORS needed since we're serving everything from the same origin

# HTML template for the main interface


@app.route("/")
def index():
    """Serve the main HTML interface"""
    try:
        # Get node configurations from Python side
        visualizer = NetworkCablingCytoscapeVisualizer()

        # Convert Python configs to JavaScript format
        node_configs = {}
        for node_type, config in visualizer.shelf_unit_configs.items():
            # Convert Python config to JavaScript format
            js_config = {
                "tray_count": config["tray_count"],
                "ports_per_tray": config["port_count"],
                "tray_layout": config["tray_layout"],
            }
            # Convert to uppercase for JavaScript (e.g., 'wh_galaxy' -> 'WH_GALAXY')
            node_configs[node_type.upper()] = js_config

        return render_template("index.html", node_configs=node_configs)

    except Exception as e:
        # Fallback to template without configs if there's an error
        return render_template("index.html", node_configs={})


@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    """Handle CSV file upload and generate visualization JSON"""
    try:
        # Check if file was uploaded
        if "csv_file" not in request.files:
            return jsonify({"success": False, "error": "No CSV file uploaded"})

        file = request.files["csv_file"]

        if file.filename == "":
            return jsonify({"success": False, "error": "No file selected"})

        if not file.filename.lower().endswith(".csv"):
            return jsonify({"success": False, "error": "File must be a CSV file"})

        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(mode="w+b", suffix=".csv", delete=False) as tmp_file:
            file.save(tmp_file.name)
            tmp_csv_path = tmp_file.name

        try:
            # Create visualizer instance and process the CSV (auto-detects format and node types)
            visualizer = NetworkCablingCytoscapeVisualizer()

            # Parse CSV file (this will auto-detect format and node types)
            connections = visualizer.parse_csv(tmp_csv_path)

            if not connections:
                return jsonify({"success": False, "error": "No valid connections found in CSV file"})

            # Generate the complete visualization data structure
            visualization_data = visualizer.generate_visualization_data()

            # Add metadata
            visualization_data["metadata"]["connection_count"] = len(connections)

            # Check for unknown node types and add to metadata
            unknown_types = visualizer.get_unknown_node_types()
            if unknown_types:
                visualization_data["metadata"]["unknown_node_types"] = unknown_types

            # Create response data
            response_data = visualization_data

            return jsonify(
                {
                    "success": True,
                    "data": response_data,
                    "message": f"Successfully processed {file.filename} with {len(connections)} connections",
                    "unknown_types": unknown_types,
                }
            )

        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_csv_path)
            except OSError:
                pass  # Ignore cleanup errors

    except Exception as e:
        error_msg = f"Error processing CSV: {str(e)}"
        return jsonify({"success": False, "error": error_msg})


@app.route("/export_cabling_descriptor", methods=["POST"])
def export_cabling_descriptor():
    """Export ClusterDescriptor from cytoscape visualization data"""
    if not EXPORT_AVAILABLE:
        return jsonify({"success": False, "error": "Export functionality not available. Missing dependencies."}), 500

    try:
        # Get cytoscape data from request
        cytoscape_data = request.get_json()
        if not cytoscape_data or "elements" not in cytoscape_data:
            return jsonify({"success": False, "error": "Invalid cytoscape data"}), 400

        # Generate textproto content
        textproto_content = export_cabling_descriptor_for_visualizer(cytoscape_data)

        # Return as plain text for download
        return Response(
            textproto_content,
            mimetype="text/plain",
            headers={"Content-Disposition": "attachment; filename=cabling_descriptor.textproto"},
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/export_deployment_descriptor", methods=["POST"])
def export_deployment_descriptor():
    """Export DeploymentDescriptor from cytoscape visualization data"""
    try:
        # Get cytoscape data from request
        cytoscape_data = request.get_json()
        if not cytoscape_data or "elements" not in cytoscape_data:
            return jsonify({"success": False, "error": "Invalid cytoscape data"}), 400

        # Generate textproto content
        textproto_content = export_deployment_descriptor_for_visualizer(cytoscape_data)

        # Return as plain text for download
        return Response(
            textproto_content,
            mimetype="text/plain",
            headers={"Content-Disposition": "attachment; filename=deployment_descriptor.textproto"},
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/generate_cabling_guide", methods=["POST"])
def generate_cabling_guide():
    """Generate CablingGuide CSV and FSD using the cabling generator"""
    import subprocess
    import tempfile
    import os
    from pathlib import Path

    try:
        # Get request data
        data = request.get_json()
        if not data or "cytoscape_data" not in data or "input_prefix" not in data:
            return jsonify({"success": False, "error": "Invalid request data"}), 400

        cytoscape_data = data["cytoscape_data"]
        input_prefix = data["input_prefix"]

        # Generate temporary files for descriptors
        with tempfile.NamedTemporaryFile(mode="w", suffix=".textproto", delete=False) as cabling_file:
            cabling_content = export_cabling_descriptor_for_visualizer(cytoscape_data)
            cabling_file.write(cabling_content)
            cabling_path = cabling_file.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".textproto", delete=False) as deployment_file:
            deployment_content = export_deployment_descriptor_for_visualizer(cytoscape_data)
            deployment_file.write(deployment_content)
            deployment_path = deployment_file.name

        try:
            # Get TT_METAL_HOME environment variable
            tt_metal_home = os.environ.get("TT_METAL_HOME")
            if not tt_metal_home:
                return jsonify({"success": False, "error": "TT_METAL_HOME environment variable not set"}), 500

            # Path to the cabling generator executable
            generator_path = os.path.join(tt_metal_home, "build", "tools", "scaleout", "run_cabling_generator")

            if not os.path.exists(generator_path):
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": f"Cabling generator not found at {generator_path}. Make sure to run ./build_metal.sh on the server first.",
                        }
                    ),
                    500,
                )

            # Create temporary output directory
            with tempfile.TemporaryDirectory() as temp_output_dir:
                # Change to the temp directory for output
                original_cwd = os.getcwd()
                os.chdir(temp_output_dir)

                try:
                    # Run the cabling generator
                    cmd = [generator_path, cabling_path, deployment_path, input_prefix]

                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

                    if result.returncode != 0:
                        return jsonify({"success": False, "error": f"Cabling generator failed: {result.stderr}"}), 500

                    # Look for generated files
                    output_dir = Path("out/scaleout")
                    cabling_guide_path = output_dir / f"cabling_guide_{input_prefix}.csv"
                    fsd_path = output_dir / f"factory_system_descriptor_{input_prefix}.textproto"

                    if not cabling_guide_path.exists() or not fsd_path.exists():
                        return jsonify({"success": False, "error": "Generated files not found"}), 500

                    # Read the generated files
                    cabling_content = cabling_guide_path.read_text()
                    fsd_content = fsd_path.read_text()

                    # Return file content directly for download
                    return jsonify(
                        {
                            "success": True,
                            "cabling_guide_content": cabling_content,
                            "cabling_guide_filename": f"cabling_guide_{input_prefix}.csv",
                            "fsd_content": fsd_content,
                            "fsd_filename": f"factory_system_descriptor_{input_prefix}.textproto",
                        }
                    )

                finally:
                    os.chdir(original_cwd)

        finally:
            # Clean up temporary descriptor files
            try:
                os.unlink(cabling_path)
                os.unlink(deployment_path)
            except:
                pass

    except subprocess.TimeoutExpired:
        return jsonify({"success": False, "error": "Cabling generator timed out"}), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/static/<path:filename>")
def static_files(filename):
    """Serve static files if needed"""
    return send_from_directory("static", filename)


@app.route("/api/node_configs", methods=["GET"])
def get_node_configs():
    """Get node configurations from Python side to ensure consistency"""
    try:
        # Create a visualizer instance to get the configurations
        visualizer = NetworkCablingCytoscapeVisualizer()

        # Convert Python configs to JavaScript format
        node_configs = {}
        for node_type, config in visualizer.shelf_unit_configs.items():
            # Convert Python config to JavaScript format
            js_config = {
                "tray_count": config["tray_count"],
                "ports_per_tray": config["port_count"],
                "tray_layout": config["tray_layout"],
            }
            # Convert to uppercase for JavaScript (e.g., 'wh_galaxy' -> 'WH_GALAXY')
            node_configs[node_type.upper()] = js_config

        return jsonify({"success": True, "node_configs": node_configs})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description="Network Cabling Visualizer Web Server")
    parser.add_argument("-p", "--port", type=int, default=5000, help="Port number to run the server on (default: 5000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind to (default: 0.0.0.0)")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (default: enabled)")
    parser.add_argument("--no-debug", dest="debug", action="store_false", help="Disable debug mode")
    parser.set_defaults(debug=True)

    args = parser.parse_args()

    print("Starting Network Cabling Visualizer Server...")
    print(f"Access the application at: http://localhost:{args.port}")
    if args.debug:
        print("Debug mode: ENABLED")
    print("Press Ctrl+C to stop the server")

    # Run Flask development server
    app.run(debug=args.debug, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
