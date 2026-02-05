#!/usr/bin/env python3
"""
Script the hosted tt-CableGen app via HTTP API: upload cabling guide CSV, get back
cabling and/or deployment descriptor (textproto).

Uses the same endpoints as the web UI:
  POST /upload_csv          – multipart form, key "csv_file" → JSON { success, data: { elements, metadata } }
  POST /export_cabling_descriptor   – JSON body { elements, metadata } → text/plain (textproto)
  POST /export_deployment_descriptor – same JSON body → text/plain (textproto)

Usage:
  # Default base URL http://localhost:5000; override with CABLEGEN_URL or --base-url
  python3 api_csv_to_descriptors.py path/to/cabling_guide.csv
  python3 api_csv_to_descriptors.py path/to/cabling_guide.csv --base-url https://cablegen.example.com
  python3 api_csv_to_descriptors.py path/to/cabling_guide.csv --cabling-out out.textproto --deployment-out deploy.textproto

  # From Python
  from api_csv_to_descriptors import csv_to_descriptors_via_api
  result = csv_to_descriptors_via_api("https://host/", csv_path="guide.csv")
  cabling_text = result["cabling_descriptor"]
  deployment_text = result["deployment_descriptor"]
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

try:
    import requests
except ImportError:
    print("This script requires the requests library. Install with: pip install requests", file=sys.stderr)
    sys.exit(1)


def csv_to_descriptors_via_api(
    base_url: str,
    csv_path: Optional[str | Path] = None,
    csv_content: Optional[str | bytes] = None,
    *,
    export_cabling: bool = True,
    export_deployment: bool = True,
    timeout: float = 60.0,
) -> Dict[str, Any]:
    """
    Call the hosted app API: upload CSV, then export cabling and/or deployment descriptor.

    Exactly one of csv_path or csv_content must be provided.

    Args:
        base_url: App base URL (e.g. https://cablegen.example.com or http://localhost:5000).
                  Trailing slash optional.
        csv_path: Path to cabling guide CSV file (read from disk).
        csv_content: CSV content as string or bytes (e.g. from another API or string buffer).
        export_cabling: Request cabling descriptor.
        export_deployment: Request deployment descriptor (may fail if no hostnames in CSV).
        timeout: Request timeout in seconds.

    Returns:
        Dict with:
          - "cabling_descriptor": str or None (textproto)
          - "deployment_descriptor": str or None (textproto)
          - "cytoscape_data": dict (elements + metadata) from upload step
          - "error": str only if upload or a requested export failed
          - "deployment_skip_reason": str if deployment was requested but failed (e.g. no hostnames)
    """
    base = base_url.rstrip("/")
    out: Dict[str, Any] = {}

    if csv_path is not None and csv_content is not None:
        out["error"] = "Provide exactly one of csv_path or csv_content"
        return out
    if csv_path is None and csv_content is None:
        out["error"] = "Provide csv_path or csv_content"
        return out

    # 1) Upload CSV
    upload_url = f"{base}/upload_csv"
    try:
        if csv_path is not None:
            path = Path(csv_path)
            if not path.is_file():
                out["error"] = f"File not found: {path}"
                return out
            with open(path, "rb") as f:
                files = {"csv_file": (path.name, f, "text/csv")}
                r = requests.post(upload_url, files=files, timeout=timeout)
        else:
            content = csv_content if isinstance(csv_content, bytes) else csv_content.encode("utf-8")
            files = {"csv_file": ("cabling_guide.csv", content, "text/csv")}
            r = requests.post(upload_url, files=files, timeout=timeout)
    except requests.RequestException as e:
        out["error"] = f"Upload request failed: {e}"
        return out

    try:
        resp = r.json()
    except Exception as e:
        out["error"] = f"Upload response not JSON: {e}"
        return out

    if not resp.get("success"):
        out["error"] = resp.get("error", "Upload failed")
        return out

    data = resp.get("data")
    if not data or "elements" not in data:
        out["error"] = "Upload response missing data.elements"
        return out

    cytoscape_data = {"elements": data["elements"], "metadata": data.get("metadata", {})}
    out["cytoscape_data"] = cytoscape_data

    # 2) Export cabling descriptor
    if export_cabling:
        try:
            r2 = requests.post(
                f"{base}/export_cabling_descriptor",
                json=cytoscape_data,
                timeout=timeout,
                headers={"Content-Type": "application/json"},
            )
            r2.raise_for_status()
            out["cabling_descriptor"] = r2.text
        except requests.RequestException as e:
            out["error"] = f"Cabling descriptor export failed: {e}"
            out["cabling_descriptor"] = None
        except Exception as e:
            out["error"] = str(e)
            out["cabling_descriptor"] = None
    else:
        out["cabling_descriptor"] = None

    # 3) Export deployment descriptor
    if export_deployment:
        try:
            r3 = requests.post(
                f"{base}/export_deployment_descriptor",
                json=cytoscape_data,
                timeout=timeout,
                headers={"Content-Type": "application/json"},
            )
            if r3.status_code != 200:
                try:
                    err = r3.json()
                    out["deployment_skip_reason"] = err.get("error", r3.text)
                except Exception:
                    out["deployment_skip_reason"] = r3.text
                out["deployment_descriptor"] = None
            else:
                out["deployment_descriptor"] = r3.text
        except requests.RequestException as e:
            out["deployment_descriptor"] = None
            out["deployment_skip_reason"] = str(e)
    else:
        out["deployment_descriptor"] = None

    return out


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Convert cabling guide CSV to descriptors via hosted tt-CableGen API.",
    )
    parser.add_argument("csv_file", type=Path, nargs="?", help="Path to cabling guide CSV")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("CABLEGEN_URL", "https://aus2-cablegen.aus2.tenstorrent.com/"),
        help="Base URL of the app (default: CABLEGEN_URL or http://localhost:5000)",
    )
    parser.add_argument("--cabling-out", type=Path, help="Write cabling descriptor to this file")
    parser.add_argument("--deployment-out", type=Path, help="Write deployment descriptor to this file")
    parser.add_argument("--cabling-only", action="store_true", help="Only request cabling descriptor")
    parser.add_argument("--deployment-only", action="store_true", help="Only request deployment descriptor")
    parser.add_argument("--timeout", type=float, default=60.0, help="Request timeout in seconds")
    args = parser.parse_args()

    if not args.csv_file or not args.csv_file.is_file():
        print("Provide a valid CSV file path.", file=sys.stderr)
        sys.exit(1)

    result = csv_to_descriptors_via_api(
        args.base_url,
        csv_path=args.csv_file,
        export_cabling=not args.deployment_only,
        export_deployment=not args.cabling_only,
        timeout=args.timeout,
    )

    if result.get("error"):
        print(result["error"], file=sys.stderr)
        sys.exit(1)

    if result.get("cabling_descriptor") and args.cabling_out:
        args.cabling_out.write_text(result["cabling_descriptor"])
        print(f"Wrote cabling descriptor to {args.cabling_out}", file=sys.stderr)
    elif result.get("cabling_descriptor") and not args.deployment_out and not args.cabling_only:
        print(result["cabling_descriptor"])

    if result.get("deployment_descriptor") and args.deployment_out:
        args.deployment_out.write_text(result["deployment_descriptor"])
        print(f"Wrote deployment descriptor to {args.deployment_out}", file=sys.stderr)
    elif result.get("deployment_descriptor") and args.deployment_only and not args.deployment_out:
        print(result["deployment_descriptor"])

    if result.get("deployment_skip_reason"):
        print(f"Deployment descriptor skipped: {result['deployment_skip_reason']}", file=sys.stderr)


if __name__ == "__main__":
    main()