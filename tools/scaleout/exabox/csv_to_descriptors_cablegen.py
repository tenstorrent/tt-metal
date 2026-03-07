#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Scriptable flow: Cabling guide CSV → Cabling descriptor and/or Deployment descriptor.

Use from the command line or import and call csv_to_descriptors() / csv_to_cabling_descriptor().

Requirements:
  - TT_METAL_HOME set and scaleout protobufs built (same as export_descriptors.py).
  - For deployment descriptor: CSV must provide hostnames (and optionally hall, aisle, rack, shelf_u)
    on the shelf rows so the visualizer can fill them; otherwise deployment export is skipped.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Project root
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from import_cabling import NetworkCablingCytoscapeVisualizer

# Export may fail if TT_METAL_HOME/protobuf not set; import only when needed
def _get_export_functions():
    try:
        from export_descriptors import (
            export_cabling_descriptor_for_visualizer,
            export_deployment_descriptor_for_visualizer,
        )
        return export_cabling_descriptor_for_visualizer, export_deployment_descriptor_for_visualizer
    except (ImportError, SystemExit) as e:
        raise RuntimeError(
            "Export descriptors not available. Set TT_METAL_HOME and build scaleout protobufs."
        ) from e


def csv_to_descriptors(
    csv_path: str | Path,
    *,
    export_cabling: bool = True,
    export_deployment: bool = True,
) -> Dict[str, Any]:
    """
    Parse a cabling guide CSV and return cabling and/or deployment descriptor textproto strings.

    Args:
        csv_path: Path to the cabling guide CSV file.
        export_cabling: If True, include cabling_descriptor in the result.
        export_deployment: If True, try to include deployment_descriptor (requires hostnames in CSV).

    Returns:
        Dict with optional keys:
          - "cabling_descriptor": str (textproto) if export_cabling and success.
          - "deployment_descriptor": str (textproto) or None if export_deployment but no hostnames / error.
          - "cytoscape_data": dict (elements + metadata) for further use.
          - "error": str only if parsing or cabling export failed.
        Deployment export is omitted (None) if hostnames are missing or export raises.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        return {"error": f"File not found: {csv_path}"}

    visualizer = NetworkCablingCytoscapeVisualizer()
    connections = visualizer.parse_csv(str(csv_path))
    if not connections:
        return {"error": "No valid connections found in CSV"}

    vis_data = visualizer.generate_visualization_data()
    cytoscape_data = {
        "elements": vis_data["elements"],
        "metadata": vis_data.get("metadata", {}),
    }

    result = {"cytoscape_data": cytoscape_data}

    if export_cabling:
        try:
            export_cabling_fn, _ = _get_export_functions()
            result["cabling_descriptor"] = export_cabling_fn(cytoscape_data)
        except Exception as e:
            result["error"] = f"Cabling descriptor export failed: {e}"
            result["cabling_descriptor"] = None
    else:
        result["cabling_descriptor"] = None

    if export_deployment:
        try:
            _, export_deployment_fn = _get_export_functions()
            result["deployment_descriptor"] = export_deployment_fn(cytoscape_data)
        except ValueError as e:
            # No hostnames or expected deployment precondition
            result["deployment_descriptor"] = None
            result["deployment_skip_reason"] = str(e)
        except Exception as e:
            result["deployment_descriptor"] = None
            result["deployment_skip_reason"] = str(e)
    else:
        result["deployment_descriptor"] = None

    return result


def csv_to_cabling_descriptor(csv_path: str | Path) -> str:
    """
    Parse cabling guide CSV and return cabling descriptor textproto string.

    Raises:
        FileNotFoundError, ValueError, RuntimeError on missing file, no connections, or export failure.
    """
    out = csv_to_descriptors(csv_path, export_cabling=True, export_deployment=False)
    if "error" in out:
        raise RuntimeError(out["error"])
    if not out.get("cabling_descriptor"):
        raise RuntimeError("No cabling descriptor produced")
    return out["cabling_descriptor"]


def csv_to_deployment_descriptor(csv_path: str | Path) -> Optional[str]:
    """
    Parse cabling guide CSV and return deployment descriptor textproto string if possible.

    Returns None if the CSV has no hostnames (or deployment export otherwise fails).
    """
    out = csv_to_descriptors(csv_path, export_cabling=False, export_deployment=True)
    return out.get("deployment_descriptor")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Convert cabling guide CSV to cabling and/or deployment descriptor (textproto)."
    )
    parser.add_argument("csv_file", type=Path, help="Input cabling guide CSV path")
    parser.add_argument(
        "--cabling-out",
        type=Path,
        default=None,
        help="Write cabling descriptor to this file (default: print to stdout if only cabling)",
    )
    parser.add_argument(
        "--deployment-out",
        type=Path,
        default=None,
        help="Write deployment descriptor to this file (optional; requires hostnames in CSV)",
    )
    parser.add_argument(
        "--cabling-only",
        action="store_true",
        help="Only produce cabling descriptor",
    )
    parser.add_argument(
        "--deployment-only",
        action="store_true",
        help="Only produce deployment descriptor (still parses CSV once)",
    )
    args = parser.parse_args()

    export_cabling = not args.deployment_only
    export_deployment = not args.cabling_only

    result = csv_to_descriptors(
        args.csv_file,
        export_cabling=export_cabling,
        export_deployment=export_deployment,
    )

    if "error" in result:
        print(result["error"], file=sys.stderr)
        sys.exit(1)

    if result.get("cabling_descriptor") and args.cabling_out:
        args.cabling_out.write_text(result["cabling_descriptor"])
        print(f"Wrote cabling descriptor to {args.cabling_out}", file=sys.stderr)
    elif result.get("cabling_descriptor") and not args.deployment_out and export_cabling:
        print(result["cabling_descriptor"])

    if result.get("deployment_descriptor") and args.deployment_out:
        args.deployment_out.write_text(result["deployment_descriptor"])
        print(f"Wrote deployment descriptor to {args.deployment_out}", file=sys.stderr)
    elif result.get("deployment_descriptor") and args.deployment_only and not args.deployment_out:
        print(result["deployment_descriptor"])

    if result.get("deployment_skip_reason") and export_deployment:
        print(f"Deployment descriptor skipped: {result['deployment_skip_reason']}", file=sys.stderr)


if __name__ == "__main__":
    main()