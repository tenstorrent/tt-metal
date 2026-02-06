#!/usr/bin/env python3
"""
CI Failure Clustering - Interactive Network Graph (Final)
Sub-node labels only appear when cluster is expanded.
Fetches OPEN issues dynamically from GitHub Projects V2 board.
"""

import json
import re
import subprocess
import sys
import time

import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
from pyvis.network import Network
from collections import Counter
import html as html_module

# ============================================================================
# GitHub Projects V2 - Dynamic Data Fetching
# ============================================================================

PROJECT_OWNER = "ebanerjeeTT"
PROJECT_NUMBER = 2


def get_gh_token():
    """Get GitHub token from gh CLI."""
    result = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: Failed to get gh auth token: {result.stderr}")
        sys.exit(1)
    return result.stdout.strip()


def fetch_project_items(token):
    """Fetch all items from the GitHub Projects V2 board, paginating through all pages."""
    url = "https://api.github.com/graphql"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    query = """
    query($login: String!, $number: Int!, $cursor: String) {
      user(login: $login) {
        projectV2(number: $number) {
          items(first: 100, after: $cursor) {
            pageInfo {
              hasNextPage
              endCursor
            }
            nodes {
              content {
                ... on Issue {
                  number
                  title
                  body
                  url
                  state
                  __typename
                }
                ... on DraftIssue {
                  __typename
                }
              }
            }
          }
        }
      }
    }
    """

    all_issues = []
    cursor = None
    page = 1

    while True:
        variables = {"login": PROJECT_OWNER, "number": PROJECT_NUMBER, "cursor": cursor}
        resp = requests.post(url, json={"query": query, "variables": variables}, headers=headers)
        resp.raise_for_status()
        result = resp.json()

        if "errors" in result:
            print(f"GraphQL errors: {result['errors']}")
            sys.exit(1)

        items_data = result["data"]["user"]["projectV2"]["items"]
        nodes = items_data["nodes"]
        print(f"  Page {page}: {len(nodes)} item(s)")

        for node in nodes:
            content = node.get("content")
            if not content:
                continue
            if content.get("__typename") != "Issue":
                continue
            if content.get("state") != "OPEN":
                continue
            all_issues.append(content)

        if not items_data["pageInfo"]["hasNextPage"]:
            break
        cursor = items_data["pageInfo"]["endCursor"]
        page += 1
        time.sleep(0.3)

    return all_issues


def extract_centroid_error(body):
    """Extract the centroid error from the ## Error Message code block."""
    match = re.search(r"## Error Message\s*```\s*(.+?)\s*```", body, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: first code block
    match = re.search(r"```\s*(.+?)\s*```", body, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def extract_all_runs(body):
    """Extract every numbered run entry from the issue body.

    Each run line looks like:
      1. [[CENTROID] timestamp (marked as ND)](url) - workflow / job (commit: abc1234...)
      2. [timestamp](url) - workflow / job (commit: abc1234...)

    Runs may optionally have an indented code block with a per-run error message.
    Returns a list of dicts with url, commit_hash, timestamp, is_nd, workflow_job, error_message.
    """
    # Matches both [CENTROID] and regular run lines
    line_pattern = r"^\d+\.\s+\[(.+)\]\(([^)]+)\)(?:\s*-\s*(.*?))?\s*(?:\(commit:\s*([a-fA-F0-9]+)\))?\s*$"

    runs = []
    lines = body.split("\n")

    current_run = None
    in_code_block = False
    code_block_lines = []

    for line in lines:
        m = re.match(line_pattern, line)
        if m:
            # Save pending code block to previous run
            if current_run and code_block_lines:
                current_run["error_message"] = "\n".join(code_block_lines).strip()
                code_block_lines = []
            in_code_block = False

            label, url, suffix, commit_hash = m.groups()

            # Only process GitHub Actions run URLs
            if not ("github.com" in url and ("actions/runs" in url or "/job/" in url)):
                current_run = None
                continue

            is_nd = "(marked as ND)" in (label or "")
            # Clean timestamp from label
            clean_label = label or ""
            if clean_label.startswith("[CENTROID]"):
                clean_label = clean_label[len("[CENTROID]") :].strip()
            clean_label = clean_label.replace("(marked as ND)", "").strip()

            workflow_job = suffix.strip() if suffix else ""

            current_run = {
                "url": url.strip(),
                "commit_hash": (commit_hash or "").strip(),
                "timestamp": clean_label,
                "is_nd": is_nd,
                "workflow_job": workflow_job,
                "error_message": "",
            }
            runs.append(current_run)

        elif current_run:
            stripped = line.strip()
            if stripped.startswith("```"):
                if in_code_block:
                    in_code_block = False
                    if code_block_lines:
                        current_run["error_message"] = "\n".join(code_block_lines).strip()
                        code_block_lines = []
                else:
                    in_code_block = True
                    code_block_lines = []
            elif in_code_block:
                code_block_lines.append(line.lstrip())

    # Handle trailing code block
    if current_run and code_block_lines:
        current_run["error_message"] = "\n".join(code_block_lines).strip()

    return runs


def parse_issues_to_failures(issues):
    """Convert raw OPEN issues into failure entries — one per run, not one per issue."""
    failures = []
    for issue in issues:
        body = issue.get("body", "") or ""
        centroid_error = extract_centroid_error(body)
        runs = extract_all_runs(body)

        if not runs:
            # No parseable runs; fall back to a single entry if we have a centroid error
            if centroid_error:
                failures.append(
                    {
                        "github_job_id": "",
                        "github_job_link": "",
                        "job_name": issue.get("title", "Unknown"),
                        "job_error_message": centroid_error,
                        "job_slack_ts": "",
                        "job_commit_hash": "",
                        "is_nd": False,
                    }
                )
            continue

        for run in runs:
            # Use the run's own error message if it has one, otherwise centroid error
            error = run["error_message"] or centroid_error
            if not error:
                continue

            job_id = ""
            if run["url"]:
                jm = re.search(r"/job/(\d+)", run["url"])
                if jm:
                    job_id = jm.group(1)

            failures.append(
                {
                    "github_job_id": job_id,
                    "github_job_link": run["url"],
                    "job_name": run["workflow_job"] or issue.get("title", "Unknown"),
                    "job_error_message": error,
                    "job_slack_ts": run["timestamp"],
                    "job_commit_hash": run["commit_hash"],
                    "is_nd": run["is_nd"],
                }
            )

    return failures


# ---- Fetch data ----
print("Fetching GitHub auth token...")
_token = get_gh_token()

print(f"Fetching OPEN issues from {PROJECT_OWNER}/projects/{PROJECT_NUMBER}...")
_raw_issues = fetch_project_items(_token)
print(f"  Found {len(_raw_issues)} OPEN issue(s)")

print("Parsing issue bodies...")
failures = parse_issues_to_failures(_raw_issues)
print(f"Loaded {len(failures)} failure entries (issues with parseable errors)")

if len(failures) < 2:
    print("ERROR: Need at least 2 failure entries to cluster. Exiting.")
    sys.exit(1)


# --- Text cleaning (matches production pipeline) ---
def clean_text(x):
    x = x.lower()
    x = re.sub(r"\s+", " ", x)
    x = re.sub(r"[^a-z0-9_\-./ ]", " ", x)
    return x.strip()


print("Cleaning error text...")
for entry in failures:
    entry["_clean"] = clean_text(entry["job_error_message"])

# Filter out very short texts (< 40 chars after cleaning)
failures = [e for e in failures if len(e["_clean"]) > 40]
print(f"  {len(failures)} entries after filtering short texts")

error_messages = [entry["_clean"] for entry in failures]

# --- TF-IDF (matches production pipeline) ---
print("Vectorizing error messages...")
_min_df = min(5, max(1, len(failures) // 10))
vectorizer = TfidfVectorizer(
    max_features=30000,
    min_df=_min_df,
    max_df=0.6,
    ngram_range=(1, 2),
    stop_words="english",
    sublinear_tf=True,
)
X = vectorizer.fit_transform(error_messages)
print(f"  TF-IDF matrix: {X.shape}")

# --- SVD dimensionality reduction ---
n_svd = min(120, X.shape[1] - 1, X.shape[0] - 1)
print(f"Reducing dimensions with SVD (n={n_svd})...")
svd = TruncatedSVD(n_components=n_svd, random_state=42)
X_reduced = svd.fit_transform(X)
print(f"  Explained variance: {svd.explained_variance_ratio_.sum():.1%}")

# --- MiniBatchKMeans k=30 (matches production pipeline) ---
n_clusters = min(30, len(failures))
print(f"Running MiniBatchKMeans clustering (k={n_clusters})...")
kmeans = MiniBatchKMeans(
    n_clusters=n_clusters,
    batch_size=min(20000, len(failures)),
    random_state=42,
    n_init="auto",
)
cluster_labels = kmeans.fit_predict(X_reduced)

# Add cluster labels to failures
for i, entry in enumerate(failures):
    entry["cluster"] = int(cluster_labels[i])


# Analyze clusters to generate meaningful names
def generate_cluster_name(cluster_entries):
    """Generate a descriptive name for a cluster based on error patterns"""
    errors = [e["job_error_message"].lower() for e in cluster_entries]
    combined = " ".join(errors)

    patterns = {
        "timeout": "Timeout Issues",
        "runner lost communication": "Runner Connection Loss",
        "weka": "Weka Filesystem Errors",
        "docker": "Docker/Container Issues",
        "fabric": "Fabric/Topology Errors",
        "circular buffer": "Memory Buffer Errors",
        "validation": "Validation Test Failures",
        "mount": "Filesystem Mount Failures",
        "client container": "Container Conflict Errors",
        "hang": "Test Hang Issues",
    }

    for pattern, name in patterns.items():
        if pattern in combined:
            return name

    job_types = [e["job_name"].split("/")[0].strip() for e in cluster_entries]
    most_common = Counter(job_types).most_common(1)[0][0]
    return f"{most_common} Failures"


print("Generating cluster names...")
cluster_names = {}
cluster_counts = Counter(cluster_labels)

# First pass: generate base names
base_names = {}
for cluster_id in range(n_clusters):
    cluster_entries = [f for f in failures if f["cluster"] == cluster_id]
    base_names[cluster_id] = generate_cluster_name(cluster_entries)

# Second pass: disambiguate duplicates with top job type
name_usage = Counter(base_names.values())
name_seen = Counter()
for cluster_id in range(n_clusters):
    base = base_names[cluster_id]
    cluster_entries = [f for f in failures if f["cluster"] == cluster_id]
    if name_usage[base] > 1:
        # Find the most common job prefix to distinguish this cluster
        job_types = [e["job_name"].split("/")[0].strip() for e in cluster_entries]
        top_job = Counter(job_types).most_common(1)[0][0]
        name_seen[base] += 1
        cluster_names[cluster_id] = f"{base} ({top_job})"
    else:
        cluster_names[cluster_id] = base
    print(f"  Cluster {cluster_id}: {cluster_names[cluster_id]} ({cluster_counts[cluster_id]} failures)")

print("\nCreating interactive network graph...")

# Create network with stable physics
net = Network(height="900px", width="100%", bgcolor="#ffffff", font_color="#333333", notebook=False, directed=False)

# Simpler, more stable physics configuration
net.set_options(
    """
{
  "physics": {
    "enabled": false
  },
  "edges": {
    "smooth": false
  },
  "interaction": {
    "hover": true,
    "tooltipDelay": 100,
    "navigationButtons": true,
    "keyboard": true,
    "zoomView": true
  }
}
"""
)

# Color palette for 30 clusters
colors = [
    "#FF6B6B",
    "#4ECDC4",
    "#45B7D1",
    "#FFA07A",
    "#98D8C8",
    "#F7DC6F",
    "#BB8FCE",
    "#85C1E2",
    "#F8B739",
    "#52B788",
    "#E74C3C",
    "#3498DB",
    "#2ECC71",
    "#9B59B6",
    "#F39C12",
    "#1ABC9C",
    "#E67E22",
    "#2980B9",
    "#27AE60",
    "#8E44AD",
    "#D35400",
    "#16A085",
    "#C0392B",
    "#7F8C8D",
    "#2C3E50",
    "#D4AC0D",
    "#A569BD",
    "#5DADE2",
    "#48C9B0",
    "#EC7063",
]


# Convert hex to RGB for transparency
def hex_to_rgba(hex_color, alpha):
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


# Pre-compute circular positions so cluster bubbles never overlap
import math

_radius = 200 * n_clusters / (2 * math.pi)  # scale ring to cluster count
_radius = max(_radius, 600)

# Add cluster center nodes
for cluster_id in range(n_clusters):
    count = cluster_counts[cluster_id]
    cluster_name = cluster_names[cluster_id]

    angle = 2 * math.pi * cluster_id / n_clusters
    cx = int(_radius * math.cos(angle))
    cy = int(_radius * math.sin(angle))

    title = f"<b>Click to expand: {cluster_name}</b><br>Total Failures: {count}"
    label = f"{cluster_name}\n({count} failures)"

    net.add_node(
        f"cluster_{cluster_id}",
        label=label,
        title=title,
        color=colors[cluster_id % len(colors)],
        size=min(30 + count, 100),
        shape="dot",
        font={"size": 14, "color": "#000000"},
        borderWidth=4,
        x=cx,
        y=cy,
        fixed=True,
    )

# Store full labels and ND status for later use
node_labels = {}
node_is_nd = {}

# Add individual failure nodes (with empty labels initially)
for i, entry in enumerate(failures):
    cluster_id = entry["cluster"]
    node_id = f"failure_{i}"

    nd_status = "Non-Deterministic (ND)" if entry["is_nd"] else "Deterministic"
    error_msg = html_module.escape(entry["job_error_message"]).replace("\n", "<br>")

    # Create the full label (to be shown when expanded)
    error_lines = entry["job_error_message"].split("\n")
    first_line = error_lines[0] if error_lines else ""
    error_preview = first_line[:50]
    if len(first_line) > 50:
        error_preview += "..."

    job_name_parts = entry["job_name"].split("/")
    job_short = job_name_parts[-1] if len(job_name_parts) > 1 else entry["job_name"]
    job_short = job_short[:25]

    full_label = f"{job_short}\n{error_preview}"

    # Store the full label and ND status for later
    node_labels[node_id] = full_label
    node_is_nd[node_id] = entry["is_nd"]

    title = f"""
    <div style='max-width: 450px; font-family: arial;'>
        <b style='font-size: 14px;'>{html_module.escape(entry['job_name'])}</b><br>
        <hr style='margin: 5px 0;'>
        <b>Status:</b> {nd_status}<br>
        <b>Commit:</b> {entry['job_commit_hash'][:8]}<br>
        <b>Cluster:</b> {cluster_names[cluster_id]}<br>
        <hr style='margin: 5px 0;'>
        <b>Error Message:</b><br>
        <div style='max-height: 250px; overflow-y: auto; font-size: 11px; padding: 8px; background: #f5f5f5; border-radius: 3px;'>
            {error_msg}
        </div>
        <hr style='margin: 5px 0;'>
        <a href='{entry['github_job_link']}' target='_blank' style='color: #0066cc; font-weight: bold;'>
            🔗 View Job on GitHub
        </a>
    </div>
    """

    error_length = len(entry["job_error_message"])
    node_size = min(12 + error_length / 40, 30)
    shape = "diamond" if entry["is_nd"] else "dot"

    # Color based on deterministic status: black for ND, red for deterministic
    node_color = "rgba(0, 0, 0, 0.08)" if entry["is_nd"] else "rgba(255, 0, 0, 0.08)"

    # Start hidden — only shown when cluster is expanded (huge perf win)
    net.add_node(
        node_id,
        label=" ",
        title=title,
        color=node_color,
        size=node_size,
        shape=shape,
        font={"size": 1, "align": "center", "color": "rgba(0,0,0,0)"},
        borderWidth=0.5,
        hidden=True,
    )

    net.add_edge(
        f"cluster_{cluster_id}",
        node_id,
        color={"color": colors[cluster_id % len(colors)], "opacity": 0.3},
        width=1.5,
        hidden=True,
    )

# Save the network
output_file = "failure_clusters_network.html"
net.save_graph(output_file)

print("Adding custom UI enhancements...")

# Read the generated HTML
with open(output_file, "r") as f:
    html_content = f.read()

# Insert custom header and functionality
header_insert = (
    """
<style>
    body { margin: 0; padding: 0; font-family: Arial, sans-serif; }
    .header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; padding: 20px; text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .header h1 { margin: 0; font-size: 28px; }
    .header p { margin: 10px 0 0 0; font-size: 14px; opacity: 0.9; }
    .instructions {
        background: #f8f9fa; padding: 15px; border-bottom: 2px solid #dee2e6;
        font-size: 13px; text-align: center; color: #495057;
    }
    .instructions b { color: #667eea; }
    .legend {
        position: absolute; top: 100px; right: 20px;
        background: white; padding: 15px; border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        font-size: 12px; z-index: 1000; max-width: 280px;
    }
    .legend h3 { margin: 0 0 10px 0; font-size: 14px; color: #333; }
    .legend-item {
        margin: 5px 0; display: flex; align-items: center;
        cursor: pointer; padding: 4px; border-radius: 4px;
        transition: background 0.2s;
    }
    .legend-item:hover { background: #f0f0f0; }
    .legend-color {
        width: 15px; height: 15px; margin-right: 8px; border-radius: 50%;
    }
    #backButton {
        position: absolute; top: 150px; left: 20px;
        background: #667eea; color: white; border: none;
        padding: 12px 24px; border-radius: 6px;
        font-size: 14px; font-weight: bold; cursor: pointer;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        z-index: 1000; display: none;
    }
    #backButton:hover { background: #5568d3; }
</style>
<div class="header">
    <h1>🔍 CI Failure Analysis Dashboard</h1>
    <p>Interactive clustering of """
    + str(len(failures))
    + """ CI failures • Click cluster nodes to expand</p>
</div>
<div class="instructions">
    <b>How to use:</b> 🖱️ Click large circles to expand and see details • Click legend to filter •
    Drag to pan • Scroll to zoom • 🔴 Red = deterministic • ⚫ Black = non-deterministic (ND) • 💎 Diamonds = ND
</div>
<button id="backButton" onclick="showAllClusters()">← Show All Clusters</button>
<div class="legend">
    <h3>📊 Click to Filter by Cluster</h3>
"""
)

for cluster_id in range(n_clusters):
    header_insert += f"""    <div class="legend-item" onclick="focusCluster({cluster_id})">
        <div class="legend-color" style="background: {colors[cluster_id]};"></div>
        <span>{cluster_names[cluster_id]} ({cluster_counts[cluster_id]})</span>
    </div>
"""

header_insert += "</div>\n"

# Replace body tag
html_content = html_content.replace("<body>", "<body>" + header_insert)

# Create JavaScript object with node labels, colors, and ND status
js_labels = "const nodeLabels = {\n"
for node_id, label in node_labels.items():
    escaped_label = label.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"').replace("'", "\\'")
    js_labels += f'  "{node_id}": "{escaped_label}",\n'
js_labels += "};\n"

js_labels += "const nodeIsND = {\n"
for node_id, is_nd in node_is_nd.items():
    js_labels += f'  "{node_id}": {str(is_nd).lower()},\n'
js_labels += "};\n"

js_labels += "const clusterColors = " + str(colors) + ";\n"

# Add JavaScript at the very end before </body>
js_code = (
    """
<script type="text/javascript">
"""
    + js_labels
    + """
(function() {
    let currentCluster = null;

    function focusCluster(clusterId) {
        currentCluster = clusterId;
        const clusterNodeId = 'cluster_' + clusterId;

        // Hide all cluster nodes except this one; unfix it so physics can act
        const allNodeIds = network.body.data.nodes.getIds();
        const nodeUpdates = [];
        allNodeIds.forEach(nodeId => {
            if (nodeId.startsWith('cluster_')) {
                if (nodeId !== clusterNodeId) {
                    nodeUpdates.push({id: nodeId, hidden: true});
                } else {
                    nodeUpdates.push({id: nodeId, fixed: false});
                }
            }
        });

        // Find edges connected to this cluster and unhide those nodes
        const edges = network.body.data.edges.get({
            filter: function(edge) {
                return edge.from === clusterNodeId || edge.to === clusterNodeId;
            }
        });
        const edgeUpdates = [];
        edges.forEach(edge => {
            const childId = edge.from === clusterNodeId ? edge.to : edge.from;
            const isND = nodeIsND[childId];
            const nodeColor = isND ? '#000000' : '#FF0000';
            nodeUpdates.push({
                id: childId,
                hidden: false,
                label: nodeLabels[childId] || "",
                color: nodeColor,
                font: {size: 9, align: 'center', color: '#000000'}
            });
            edgeUpdates.push({id: edge.id, hidden: false});
        });

        network.body.data.nodes.update(nodeUpdates);
        network.body.data.edges.update(edgeUpdates);

        // Enable physics briefly so children spread out
        network.setOptions({physics: {
            enabled: true,
            stabilization: false,
            barnesHut: {gravitationalConstant: -8000, springLength: 150, springConstant: 0.04, damping: 0.9}
        }});

        document.getElementById('backButton').style.display = 'block';
        setTimeout(() => {
            network.setOptions({physics: {enabled: false}});
            network.fit({animation: {duration: 600, easingFunction: 'easeInOutQuad'}});
        }, 800);
    }

    function showAllClusters() {
        // Hide all failure nodes and edges, restore all cluster nodes as fixed
        const allNodeIds = network.body.data.nodes.getIds();
        const nodeUpdates = [];
        allNodeIds.forEach(nodeId => {
            if (nodeId.startsWith('failure_')) {
                nodeUpdates.push({id: nodeId, hidden: true});
            } else {
                nodeUpdates.push({id: nodeId, hidden: false, fixed: true});
            }
        });
        network.body.data.nodes.update(nodeUpdates);

        const allEdges = network.body.data.edges.get();
        const edgeUpdates = allEdges.map(e => ({id: e.id, hidden: true}));
        network.body.data.edges.update(edgeUpdates);

        network.setOptions({physics: {enabled: false}});
        document.getElementById('backButton').style.display = 'none';
        setTimeout(() => {
            network.fit({animation: {duration: 600, easingFunction: 'easeInOutQuad'}});
        }, 100);
    }

    window.focusCluster = focusCluster;
    window.showAllClusters = showAllClusters;

    network.on("click", function(params) {
        if (params.nodes.length > 0) {
            const nodeId = params.nodes[0];
            if (nodeId.startsWith('cluster_')) {
                const clusterId = parseInt(nodeId.split('_')[1]);
                focusCluster(clusterId);
            }
        }
    });
})();
</script>
"""
)

html_content = html_content.replace("</body>", js_code + "</body>")

with open(output_file, "w") as f:
    f.write(html_content)

print(f"\n✓ Final interactive graph saved to: {output_file}")
print(f"\n📊 Features:")
print(f"  ✓ Clean initial view (sub-node labels hidden)")
print(f"  ✓ Click cluster nodes to expand and reveal details")
print(f"  ✓ Labels appear only when cluster is expanded")
print(f"  ✓ Stable physics, smooth animations")
print(f"  ✓ Click legend to filter clusters")
print("\n" + "=" * 80)
print("Perfect for your meeting! Open 'failure_clusters_network.html'")
print("=" * 80)
