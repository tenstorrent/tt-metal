# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Circle
import math


def draw(outpath="exabox_logical.png"):
    # --- style tuned toward OG ---
    node_fill = "#bcdbe8"
    node_edge = "#86a8ba"
    dot_color = "#2f6fb1"
    edge_color = "#4b4b4b"

    inter_lw = 1.25
    dash = (0, (3, 5))
    dash_lw = 1.15

    # geometry
    d = 4.05
    cluster_radius = 1.8

    G = nx.Graph()
    pos = {}

    # 5 Galaxy centers in a pentagon (top, then clockwise)
    n_galaxies = 5
    centers = {}
    for i in range(n_galaxies):
        angle = math.radians(90 - i * 72)
        centers[i] = (d * math.cos(angle), d * math.sin(angle))

    # Add nodes
    for gid, (cx, cy) in centers.items():
        nid = f"g{gid}"
        G.add_node(nid, label=str(gid))
        pos[nid] = (cx, cy)

    # All-to-all edges (10 connections)
    for i in range(n_galaxies):
        for j in range(i + 1, n_galaxies):
            G.add_edge(f"g{i}", f"g{j}", kind="inter")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.axis("off")

    # dashed cluster circles + titles
    for gid, (cx, cy) in centers.items():
        ax.add_patch(
            Circle(
                (cx, cy),
                cluster_radius,
                fill=False,
                linestyle=dash,
                linewidth=dash_lw,
                alpha=0.70,
                edgecolor=node_edge,
                zorder=0,
            )
        )
        ax.text(
            cx,
            cy + cluster_radius + 0.35,
            f"Galaxy {gid}",
            ha="center",
            va="bottom",
            fontsize=12,
            color="#2b2b2b",
            zorder=5,
        )

    # inter-galaxy edges with boundary dots
    boundary_dots = []
    for i in range(n_galaxies):
        for j in range(i + 1, n_galaxies):
            x1, y1 = centers[i]
            x2, y2 = centers[j]

            # direction vector
            dx, dy = x2 - x1, y2 - y1
            dist = math.sqrt(dx**2 + dy**2)
            dx, dy = dx / dist, dy / dist

            # edge endpoints on circle boundaries
            sx, sy = x1 + cluster_radius * dx, y1 + cluster_radius * dy
            ex, ey = x2 - cluster_radius * dx, y2 - cluster_radius * dy

            ax.plot([sx, ex], [sy, ey], color=edge_color, linewidth=inter_lw, alpha=0.9, zorder=1)
            boundary_dots.extend([(sx, sy), (ex, ey)])

    # nodes + labels on top
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=1250,
        node_color=node_fill,
        edgecolors=node_edge,
        linewidths=1.35,
    )
    labels = {n: G.nodes[n]["label"] for n in G.nodes}
    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        ax=ax,
        font_size=12,
        font_weight="normal",
        font_color="#2b2b2b",
    )

    # boundary dots
    xs = [p[0] for p in boundary_dots]
    ys = [p[1] for p in boundary_dots]
    ax.scatter(xs, ys, s=18, color=dot_color, edgecolors=dot_color, zorder=6)

    lim = d + cluster_radius + 1.2
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    plt.tight_layout(pad=0.2)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    draw("exabox_logical.png")
