#!/usr/bin/env python3
"""
CI Failure Clustering - Interactive Network Graph
Creates a force-directed graph with expandable cluster nodes
"""

import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from pyvis.network import Network
from collections import Counter
import re

# Load the JSON data
print("Loading failure data...")
with open("build_temp.json", "r") as f:
    failures = json.load(f)

print(f"Loaded {len(failures)} failure entries")

# Extract error messages
error_messages = [entry["job_error_message"] for entry in failures]

print("Vectorizing error messages...")
vectorizer = TfidfVectorizer(max_features=500, stop_words="english", ngram_range=(1, 2), min_df=2)
X = vectorizer.fit_transform(error_messages)

print("Running K-means clustering (k=10)...")
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X)

# Add cluster labels to failures
for i, entry in enumerate(failures):
    entry["cluster"] = int(cluster_labels[i])


# Analyze clusters to generate meaningful names
def generate_cluster_name(cluster_entries):
    """Generate a descriptive name for a cluster based on error patterns"""
    errors = [e["job_error_message"].lower() for e in cluster_entries]
    combined = " ".join(errors)

    # Common patterns to look for
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

    # Fallback: use most common job type
    job_types = [e["job_name"].split("/")[0].strip() for e in cluster_entries]
    most_common = Counter(job_types).most_common(1)[0][0]
    return f"{most_common} Failures"


print("Generating cluster names...")
cluster_names = {}
cluster_counts = Counter(cluster_labels)

for cluster_id in range(10):
    cluster_entries = [f for f in failures if f["cluster"] == cluster_id]
    cluster_names[cluster_id] = generate_cluster_name(cluster_entries)
    print(f"  Cluster {cluster_id}: {cluster_names[cluster_id]} ({len(cluster_entries)} failures)")

print("\nCreating interactive network graph...")

# Create network
net = Network(height="900px", width="100%", bgcolor="#ffffff", font_color="#333333", notebook=False, directed=False)

# Configure physics for force-directed layout
net.set_options(
    """
{
  "physics": {
    "enabled": true,
    "forceAtlas2Based": {
      "gravitationalConstant": -50,
      "centralGravity": 0.01,
      "springLength": 200,
      "springConstant": 0.08,
      "damping": 0.4,
      "avoidOverlap": 1
    },
    "maxVelocity": 50,
    "minVelocity": 0.1,
    "solver": "forceAtlas2Based",
    "timestep": 0.35,
    "stabilization": {
      "enabled": true,
      "iterations": 1000,
      "updateInterval": 25
    }
  },
  "interaction": {
    "hover": true,
    "tooltipDelay": 100,
    "navigationButtons": true,
    "keyboard": true,
    "zoomView": true
  },
  "nodes": {
    "font": {
      "size": 14,
      "face": "arial"
    }
  },
  "edges": {
    "smooth": {
      "enabled": true,
      "type": "continuous"
    }
  }
}
"""
)

# Color palette for clusters
colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E2", "#F8B739", "#52B788"]

# Add cluster center nodes (large, labeled)
for cluster_id in range(10):
    count = cluster_counts[cluster_id]
    cluster_name = cluster_names[cluster_id]

    title = f"<b>{cluster_name}</b><br>Total Failures: {count}<br><br>Click and drag to explore"

    net.add_node(
        f"cluster_{cluster_id}",
        label=f"{cluster_name}\n({count} failures)",
        title=title,
        color=colors[cluster_id],
        size=40 + count * 2,  # Scale by number of failures
        shape="dot",
        font={"size": 18, "color": "#000000", "bold": True},
        borderWidth=3,
        borderWidthSelected=5,
    )

# Add individual failure nodes connected to their clusters
for i, entry in enumerate(failures):
    cluster_id = entry["cluster"]

    # Create node ID
    node_id = f"failure_{i}"

    # Prepare hover text with all details
    nd_status = "Non-Deterministic (ND)" if entry["is_nd"] else "Deterministic"
    error_msg = entry["job_error_message"].replace("\n", "<br>")

    # Truncate error for label, full text in hover
    error_preview = entry["job_error_message"][:80].replace("\n", " ")
    if len(entry["job_error_message"]) > 80:
        error_preview += "..."

    title = f"""
    <div style='max-width: 400px; font-family: arial;'>
        <b style='font-size: 14px;'>{entry['job_name']}</b><br>
        <hr style='margin: 5px 0;'>
        <b>Status:</b> {nd_status}<br>
        <b>Commit:</b> {entry['job_commit_hash'][:8]}<br>
        <b>Cluster:</b> {cluster_names[cluster_id]}<br>
        <hr style='margin: 5px 0;'>
        <b>Error Message:</b><br>
        <div style='max-height: 200px; overflow-y: auto; font-size: 11px; padding: 5px; background: #f5f5f5; border-radius: 3px;'>
            {error_msg}
        </div>
        <hr style='margin: 5px 0;'>
        <a href='{entry['github_job_link']}' target='_blank' style='color: #0066cc;'>
            🔗 View Job on GitHub
        </a>
    </div>
    """

    # Node size scales with error message length (normalized)
    error_length = len(entry["job_error_message"])
    node_size = min(8 + error_length / 50, 25)  # Size between 8 and 25

    # Node color matches cluster but lighter
    node_color = colors[cluster_id]

    # Shape based on ND status
    shape = "diamond" if entry["is_nd"] else "dot"

    # Label shows job name (truncated)
    label_parts = entry["job_name"].split("/")
    label = label_parts[-1][:20] if len(label_parts) > 1 else entry["job_name"][:20]

    net.add_node(
        node_id,
        label=label,
        title=title,
        color={"background": node_color, "border": "#333333"},
        size=node_size,
        shape=shape,
        font={"size": 10},
        borderWidth=1,
    )

    # Connect to cluster center
    net.add_edge(f"cluster_{cluster_id}", node_id, color={"color": node_color, "opacity": 0.3}, width=1)

# Save the network
output_file = "failure_clusters_network.html"
net.save_graph(output_file)

# Add custom header and instructions to the HTML
with open(output_file, "r") as f:
    html_content = f.read()

# Insert custom CSS and header
header_html = """
<style>
    body {
        margin: 0;
        padding: 0;
        font-family: Arial, sans-serif;
    }
    .header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .header h1 {
        margin: 0;
        font-size: 28px;
    }
    .header p {
        margin: 10px 0 0 0;
        font-size: 14px;
        opacity: 0.9;
    }
    .instructions {
        background: #f8f9fa;
        padding: 15px;
        border-bottom: 2px solid #dee2e6;
        font-size: 13px;
        text-align: center;
        color: #495057;
    }
    .instructions b {
        color: #667eea;
    }
    .legend {
        position: absolute;
        top: 100px;
        right: 20px;
        background: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        font-size: 12px;
        z-index: 1000;
        max-width: 250px;
    }
    .legend h3 {
        margin: 0 0 10px 0;
        font-size: 14px;
        color: #333;
    }
    .legend-item {
        margin: 5px 0;
        display: flex;
        align-items: center;
    }
    .legend-color {
        width: 15px;
        height: 15px;
        margin-right: 8px;
        border-radius: 50%;
    }
</style>
<div class="header">
    <h1>🔍 CI Failure Analysis Dashboard</h1>
    <p>Interactive clustering of 212 CI failures across 10 categories</p>
</div>
<div class="instructions">
    <b>How to use:</b>
    🖱️ Click and drag to pan • 🔍 Scroll to zoom •
    ⭕ Large circles = failure categories • 💎 Diamonds = non-deterministic failures •
    ⚪ Small circles = deterministic failures •
    Hover over any node for details
</div>
<div class="legend">
    <h3>📊 Cluster Categories</h3>
"""

for cluster_id in range(10):
    header_html += f"""
    <div class="legend-item">
        <div class="legend-color" style="background: {colors[cluster_id]};"></div>
        <span>{cluster_names[cluster_id]} ({cluster_counts[cluster_id]})</span>
    </div>
"""

header_html += "</div>"

# Insert header after <body> tag
html_content = html_content.replace("<body>", "<body>" + header_html)

with open(output_file, "w") as f:
    f.write(html_content)

print(f"\n✓ Interactive graph saved to: {output_file}")
print(f"  Open this file in your browser to explore the failure clusters")
print(f"\n📊 Graph includes:")
print(f"  • 10 large cluster nodes (categories)")
print(f"  • 212 individual failure nodes")
print(f"  • Force-directed layout for natural grouping")
print(f"  • Hover tooltips with full details and clickable links")
print(f"  • Diamond shapes for non-deterministic failures")
print(f"  • Node sizes scale with error message length")

print("\n" + "=" * 80)
print("Done! Open 'failure_clusters_network.html' in your browser.")
print("=" * 80)
