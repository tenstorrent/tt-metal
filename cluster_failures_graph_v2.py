#!/usr/bin/env python3
"""
CI Failure Clustering - Interactive Network Graph v2
Enhanced with click-to-expand functionality and visible error text on nodes
"""

import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from pyvis.network import Network
from collections import Counter
import html

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

for cluster_id in range(10):
    cluster_entries = [f for f in failures if f["cluster"] == cluster_id]
    cluster_names[cluster_id] = generate_cluster_name(cluster_entries)
    print(f"  Cluster {cluster_id}: {cluster_names[cluster_id]} ({len(cluster_entries)} failures)")

print("\nCreating interactive network graph...")

# Create network with slower physics
net = Network(height="900px", width="100%", bgcolor="#ffffff", font_color="#333333", notebook=False, directed=False)

# Configure physics for slower, more stable movement
net.set_options(
    """
{
  "physics": {
    "enabled": true,
    "forceAtlas2Based": {
      "gravitationalConstant": -30,
      "centralGravity": 0.005,
      "springLength": 150,
      "springConstant": 0.02,
      "damping": 0.8,
      "avoidOverlap": 0.5
    },
    "maxVelocity": 20,
    "minVelocity": 0.5,
    "solver": "forceAtlas2Based",
    "timestep": 0.5,
    "stabilization": {
      "enabled": true,
      "iterations": 500,
      "updateInterval": 50
    }
  },
  "interaction": {
    "hover": true,
    "tooltipDelay": 100,
    "navigationButtons": true,
    "keyboard": true,
    "zoomView": true,
    "dragNodes": true,
    "dragView": true
  },
  "nodes": {
    "font": {
      "size": 12,
      "face": "arial",
      "multi": "html"
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

# Add cluster center nodes (large, labeled, clickable)
for cluster_id in range(10):
    count = cluster_counts[cluster_id]
    cluster_name = cluster_names[cluster_id]

    title = f"<b>Click to expand: {cluster_name}</b><br>Total Failures: {count}"

    # Multi-line label with name and count
    label = f"{cluster_name}\n({count} failures)"

    net.add_node(
        f"cluster_{cluster_id}",
        label=label,
        title=title,
        color=colors[cluster_id],
        size=50 + count * 2,
        shape="dot",
        font={"size": 16, "color": "#000000", "bold": True, "multi": "html"},
        borderWidth=4,
        borderWidthSelected=6,
        group=f"cluster_{cluster_id}",
    )

# Add individual failure nodes with visible error text
for i, entry in enumerate(failures):
    cluster_id = entry["cluster"]
    node_id = f"failure_{i}"

    # Prepare hover text with all details
    nd_status = "Non-Deterministic (ND)" if entry["is_nd"] else "Deterministic"
    error_msg = html.escape(entry["job_error_message"]).replace("\n", "<br>")

    # Truncate error for display on node
    error_lines = entry["job_error_message"].split("\n")
    first_line = error_lines[0] if error_lines else ""
    error_preview = first_line[:60]
    if len(first_line) > 60:
        error_preview += "..."

    # Job name for label
    job_name_parts = entry["job_name"].split("/")
    job_short = job_name_parts[-1] if len(job_name_parts) > 1 else entry["job_name"]
    job_short = job_short[:30]

    # Create multi-line label with job name and error preview
    node_label = f"{job_short}\n{error_preview}"

    title = f"""
    <div style='max-width: 450px; font-family: arial;'>
        <b style='font-size: 14px;'>{html.escape(entry['job_name'])}</b><br>
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

    # Node size scales with error message length
    error_length = len(entry["job_error_message"])
    node_size = min(12 + error_length / 40, 30)

    # Shape based on ND status
    shape = "diamond" if entry["is_nd"] else "dot"

    net.add_node(
        node_id,
        label=node_label,
        title=title,
        color={"background": colors[cluster_id], "border": "#333333"},
        size=node_size,
        shape=shape,
        font={"size": 9, "multi": "html", "align": "center"},
        borderWidth=1,
        group=f"cluster_{cluster_id}",
    )

    # Connect to cluster center
    net.add_edge(f"cluster_{cluster_id}", node_id, color={"color": colors[cluster_id], "opacity": 0.3}, width=1.5)

# Save the network
output_file = "failure_clusters_network.html"
net.save_graph(output_file)

# Add custom JavaScript for click-to-expand functionality
with open(output_file, "r") as f:
    html_content = f.read()

# Insert custom CSS and header with expand functionality
custom_code = """
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
        max-width: 280px;
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
        cursor: pointer;
        padding: 4px;
        border-radius: 4px;
        transition: background 0.2s;
    }
    .legend-item:hover {
        background: #f0f0f0;
    }
    .legend-color {
        width: 15px;
        height: 15px;
        margin-right: 8px;
        border-radius: 50%;
    }
    #backButton {
        position: absolute;
        top: 150px;
        left: 20px;
        background: #667eea;
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 6px;
        font-size: 14px;
        font-weight: bold;
        cursor: pointer;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        z-index: 1000;
        display: none;
    }
    #backButton:hover {
        background: #5568d3;
    }
</style>
<div class="header">
    <h1>🔍 CI Failure Analysis Dashboard</h1>
    <p>Interactive clustering of 212 CI failures • Click cluster nodes to expand</p>
</div>
<div class="instructions">
    <b>How to use:</b>
    🖱️ Click large circles to expand category • Click "Show All" to reset •
    Drag to pan • Scroll to zoom •
    💎 Diamonds = non-deterministic • ⚪ Circles = deterministic
</div>
<button id="backButton" onclick="showAllClusters()">← Show All Clusters</button>
<div class="legend">
    <h3>📊 Click to Filter by Cluster</h3>
"""

for cluster_id in range(10):
    custom_code += f"""
    <div class="legend-item" onclick="focusCluster({cluster_id})">
        <div class="legend-color" style="background: {colors[cluster_id]};"></div>
        <span>{cluster_names[cluster_id]} ({cluster_counts[cluster_id]})</span>
    </div>
"""

custom_code += """</div>
<script type="text/javascript">
    let currentCluster = null;
    let allNodes = null;

    // Wait for network to be ready
    window.addEventListener('load', function() {
        setTimeout(function() {
            allNodes = network.body.data.nodes.get();

            // Add click event to cluster nodes
            network.on("click", function(params) {
                if (params.nodes.length > 0) {
                    const nodeId = params.nodes[0];
                    if (nodeId.startsWith('cluster_')) {
                        const clusterId = parseInt(nodeId.split('_')[1]);
                        focusCluster(clusterId);
                    }
                }
            });

            // Slow down physics after initial stabilization
            network.on("stabilizationIterationsDone", function() {
                network.setOptions({
                    physics: {
                        enabled: true,
                        forceAtlas2Based: {
                            damping: 0.95
                        }
                    }
                });
            });
        }, 100);
    });

    function focusCluster(clusterId) {
        currentCluster = clusterId;
        const clusterNodeId = 'cluster_' + clusterId;

        // Get all nodes
        const allNodeIds = network.body.data.nodes.getIds();

        // Hide all nodes except those in the selected cluster
        allNodeIds.forEach(nodeId => {
            if (nodeId.startsWith('cluster_') && nodeId !== clusterNodeId) {
                network.body.data.nodes.update({id: nodeId, hidden: true});
            } else if (nodeId.startsWith('failure_')) {
                const node = network.body.data.nodes.get(nodeId);
                if (node.group !== 'cluster_' + clusterId) {
                    network.body.data.nodes.update({id: nodeId, hidden: true});
                }
            }
        });

        // Show back button
        document.getElementById('backButton').style.display = 'block';

        // Focus on the cluster
        setTimeout(function() {
            network.fit({
                nodes: [clusterNodeId],
                animation: {
                    duration: 1000,
                    easingFunction: 'easeInOutQuad'
                }
            });
        }, 100);
    }

    function showAllClusters() {
        currentCluster = null;

        // Show all nodes
        const allNodeIds = network.body.data.nodes.getIds();
        allNodeIds.forEach(nodeId => {
            network.body.data.nodes.update({id: nodeId, hidden: false});
        });

        // Hide back button
        document.getElementById('backButton').style.display = 'none';

        // Fit to show all
        setTimeout(function() {
            network.fit({
                animation: {
                    duration: 1000,
                    easingFunction: 'easeInOutQuad'
                }
            });
        }, 100);
    }
</script>
"""

# Insert after <body> tag
html_content = html_content.replace("<body>", "<body>" + custom_code)

with open(output_file, "w") as f:
    f.write(html_content)

print(f"\n✓ Enhanced interactive graph saved to: {output_file}")
print(f"\n📊 New features:")
print(f"  ✓ Click any large cluster node to expand and focus")
print(f"  ✓ Error messages visible on failure nodes")
print(f"  ✓ Slower, more stable node movement")
print(f"  ✓ 'Show All Clusters' button to reset view")
print(f"  ✓ Clickable legend to filter by cluster")
print(f"  ✓ Smooth zoom animations")

print("\n" + "=" * 80)
print("Done! Open 'failure_clusters_network.html' in your browser.")
print("=" * 80)
