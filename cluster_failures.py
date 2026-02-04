#!/usr/bin/env python3
"""
CI Failure Clustering Script
Clusters error messages using K-means and creates an interactive visualization
"""

import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter

# Load the JSON data
print("Loading failure data...")
with open("build_temp.json", "r") as f:
    failures = json.load(f)

print(f"Loaded {len(failures)} failure entries")

# Extract error messages and metadata
error_messages = []
job_names = []
job_links = []
is_nd_list = []
commit_hashes = []

for entry in failures:
    error_messages.append(entry["job_error_message"])
    job_names.append(entry["job_name"])
    job_links.append(entry["github_job_link"])
    is_nd_list.append(entry["is_nd"])
    commit_hashes.append(entry["job_commit_hash"][:8])  # Short hash

print("Vectorizing error messages...")
# Vectorize error messages using TF-IDF
vectorizer = TfidfVectorizer(max_features=500, stop_words="english", ngram_range=(1, 2), min_df=2)
X = vectorizer.fit_transform(error_messages)

print("Running K-means clustering (k=10)...")
# Perform K-means clustering with 10 clusters
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X)

# Add cluster labels to our data
for i, entry in enumerate(failures):
    entry["cluster"] = int(cluster_labels[i])

print("Reducing dimensions for visualization...")
# Reduce to 2D for visualization using PCA
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X.toarray())

# Count cluster sizes
cluster_counts = Counter(cluster_labels)
print("\nCluster sizes:")
for cluster_id in sorted(cluster_counts.keys()):
    print(f"  Cluster {cluster_id}: {cluster_counts[cluster_id]} failures")

# Create hover text with job details
hover_texts = []
for i in range(len(failures)):
    nd_status = "ND" if is_nd_list[i] else "Deterministic"
    hover_text = (
        f"<b>Cluster {cluster_labels[i]}</b><br>"
        f"<b>{job_names[i]}</b><br>"
        f"Status: {nd_status}<br>"
        f"Commit: {commit_hashes[i]}<br>"
        f"<br>"
        f"Error: {error_messages[i][:200]}{'...' if len(error_messages[i]) > 200 else ''}<br>"
        f"<br>"
        f"<a href='{job_links[i]}'>View Job</a>"
    )
    hover_texts.append(hover_text)

# Create color palette for clusters
colors = px.colors.qualitative.Set3[:10]

print("Generating interactive visualization...")
# Create the scatter plot
fig = go.Figure()

for cluster_id in range(10):
    mask = cluster_labels == cluster_id
    fig.add_trace(
        go.Scatter(
            x=X_2d[mask, 0],
            y=X_2d[mask, 1],
            mode="markers",
            name=f"Cluster {cluster_id} ({cluster_counts[cluster_id]} failures)",
            marker=dict(size=10, color=colors[cluster_id], line=dict(width=0.5, color="white"), opacity=0.8),
            text=[hover_texts[i] for i in range(len(hover_texts)) if mask[i]],
            hovertemplate="%{text}<extra></extra>",
            showlegend=True,
        )
    )

# Update layout
fig.update_layout(
    title={
        "text": "CI Failure Clusters - K-means (k=10)<br><sub>Interactive visualization of 212 failure entries</sub>",
        "x": 0.5,
        "xanchor": "center",
        "font": {"size": 20},
    },
    xaxis_title="PCA Component 1",
    yaxis_title="PCA Component 2",
    hovermode="closest",
    width=1400,
    height=900,
    showlegend=True,
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.8)"),
    plot_bgcolor="#f8f9fa",
)

# Save the interactive visualization
output_file = "failure_clusters.html"
fig.write_html(output_file)
print(f"\n✓ Visualization saved to: {output_file}")
print(f"  Open this file in your browser to explore the clusters")

# Save clustered data back to JSON for reference
output_json = "failures_clustered.json"
with open(output_json, "w") as f:
    json.dump(failures, f, indent=2)
print(f"✓ Clustered data saved to: {output_json}")

# Print summary of each cluster with sample errors
print("\n" + "=" * 80)
print("CLUSTER SUMMARY")
print("=" * 80)
for cluster_id in range(10):
    cluster_entries = [f for f in failures if f["cluster"] == cluster_id]
    print(f"\n--- Cluster {cluster_id} ({len(cluster_entries)} failures) ---")

    # Show first 2 representative errors
    for idx, entry in enumerate(cluster_entries[:2]):
        error_preview = entry["job_error_message"][:150].replace("\n", " ")
        print(f"  Example {idx+1}: {error_preview}...")

    # Show most common job types in this cluster
    job_types = [e["job_name"].split("/")[0].strip() for e in cluster_entries]
    common_jobs = Counter(job_types).most_common(3)
    print(f"  Common job types: {', '.join([f'{j} ({c})' for j, c in common_jobs])}")

print("\n" + "=" * 80)
print("Done! Open 'failure_clusters.html' in your browser.")
print("=" * 80)
