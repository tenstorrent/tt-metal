#!/usr/bin/env python3
"""Generate diagram 1: Current Flow (Pre-PR 36174) - Strictly Vertical Layout"""

import graphviz

# Create a new directed graph
dot = graphviz.Digraph(comment="Current Flow Diagram", format="png")
dot.attr(rankdir="TB", nodesep="0.3", ranksep="0.6", compound="true")
dot.attr("node", fontsize="11", fontname="Arial")
dot.attr("edge", fontsize="10", fontname="Arial")

# --- Nodes ---

dot.node("User", "User", shape="ellipse", style="filled", fillcolor="#FFE5B4", width="1.5")
dot.node("SLURM", "SLURM\nScheduler", shape="box", style="rounded,filled", fillcolor="#90EE90", width="1.5")

dot.node(
    "RankBindings",
    "rank_bindings.yaml",
    shape="note",
    style="filled",
    fillcolor="#FFB6C1",
    fontsize="10",
    fontweight="bold",
)
dot.node("MGD", "mesh_graph_descriptor.textproto", shape="note", style="filled", fillcolor="#FFB6C1", fontsize="10")
dot.node("Rankfile", "rankfile.txt", shape="note", style="filled", fillcolor="#FFB6C1", fontsize="10")
dot.node("TTVisible", "TT_VISIBLE_DEVICES", shape="note", style="filled", fillcolor="#FFB6C1", fontsize="10")

dot.node("TTRun", "tt-run", shape="box", style="rounded,filled", fillcolor="#87CEEB", width="1.2")

# Multiple MPI Processes
dot.node("MPI1", "MPI\nProcess 1", shape="box", style="rounded,filled", fillcolor="#87CEEB", width="1.2")
dot.node("MPI2", "MPI\nProcess 2", shape="box", style="rounded,filled", fillcolor="#87CEEB", width="1.2")
dot.node("MPI3", "MPI\nProcess 3", shape="box", style="rounded,filled", fillcolor="#87CEEB", width="1.2")

# Multiple Control Planes
with dot.subgraph(name="cluster_Runtime") as runtime:
    runtime.attr(label="Runtime / Control Plane", style="filled", color="#F0E0FF", fillcolor="#F0E0FF")
    dot.node("CP1", "Control Plane", shape="box", style="filled", fillcolor="#DDA0DD", width="1.3")
    dot.node("CP2", "Control Plane", shape="box", style="filled", fillcolor="#DDA0DD", width="1.3")
    dot.node("CP3", "Control Plane", shape="box", style="filled", fillcolor="#DDA0DD", width="1.3")

# Multiple TT Devices
dot.node("Device1", "TT Device 1", shape="box", style="filled", fillcolor="#F0E68C", width="1.2")
dot.node("Device2", "TT Device 2", shape="box", style="filled", fillcolor="#F0E68C", width="1.2")
dot.node("Device3", "TT Device 3", shape="box", style="filled", fillcolor="#F0E68C", width="1.2")

# --- Strict Vertical Ranks ---

# Rank 1: User
with dot.subgraph() as r1:
    r1.attr(rank="same")
    r1.node("User")

# Rank 2: SLURM
with dot.subgraph() as r2:
    r2.attr(rank="same")
    r2.node("SLURM")

# Rank 3: Configuration Files
with dot.subgraph() as r3:
    r3.attr(rank="same")
    r3.node("RankBindings")
    r3.node("MGD")
    r3.node("Rankfile")
    r3.node("TTVisible")

# Rank 4: tt-run
with dot.subgraph() as r4:
    r4.attr(rank="same")
    r4.node("TTRun")

# Rank 5: MPI Processes (Multiple)
with dot.subgraph() as r5:
    r5.attr(rank="same")
    r5.node("MPI1")
    r5.node("MPI2")
    r5.node("MPI3")

# Rank 6: Control Plane (Multiple)
with dot.subgraph() as r6:
    r6.attr(rank="same")
    r6.node("CP1")
    r6.node("CP2")
    r6.node("CP3")

# Rank 7: Devices (Bottom)
with dot.subgraph() as r7:
    r7.attr(rank="same")
    r7.node("Device1")
    r7.node("Device2")
    r7.node("Device3")

# --- Vertical Alignment (Invisible) ---
dot.edge("User", "SLURM", style="invis")
dot.edge("SLURM", "RankBindings", style="invis")
dot.edge("RankBindings", "TTRun", style="invis")
dot.edge("TTRun", "MPI1", style="invis")
dot.edge("MPI1", "CP1", style="invis")
dot.edge("CP1", "Device1", style="invis")

# Keep MPI processes aligned
dot.edge("MPI1", "MPI2", style="invis")
dot.edge("MPI2", "MPI3", style="invis")
dot.edge("CP1", "CP2", style="invis")
dot.edge("CP2", "CP3", style="invis")
dot.edge("Device1", "Device2", style="invis")
dot.edge("Device2", "Device3", style="invis")

# --- Edges ---

# 1. Allocation
dot.edge("User", "SLURM", label="1. salloc")
dot.edge("SLURM", "User", label="Allocated", style="dashed", constraint="false")

# 2. Configuration
dot.edge("User", "RankBindings", label="2. Create Manual")
dot.edge("User", "MGD", label="2. Create Manual")
dot.edge("User", "Rankfile", label="2. Create Manual")
dot.edge("User", "TTVisible", label="2. Create Manual")
dot.edge("RankBindings", "MGD", style="invis")  # Keep files close
dot.edge("MGD", "Rankfile", style="invis")
dot.edge("Rankfile", "TTVisible", style="invis")

# 3. Execution
dot.edge("User", "TTRun", label="3. Exec", constraint="false")
dot.edge("RankBindings", "TTRun", label="--rank-binding", color="#C71585", penwidth="2.0")
dot.edge("Rankfile", "TTRun", label="--rankfile", color="#C71585", penwidth="2.0")
dot.edge("TTVisible", "TTRun", label="env var", color="#C71585", penwidth="2.0")
dot.edge("MGD", "TTRun", label="MGD path", color="#C71585", penwidth="2.0")

# 4. Launch (to all MPI processes)
dot.edge("TTRun", "MPI1", label="4. Launch")
dot.edge("TTRun", "MPI2", label="Launch")
dot.edge("TTRun", "MPI3", label="Launch")

# 5. Discovery and Mapping (each MPI to its CP)
dot.edge("MPI1", "CP1", label="5. Discovery\nand Mapping")
dot.edge("MPI2", "CP2", label="Discovery\nand Mapping")
dot.edge("MPI3", "CP3", label="Discovery\nand Mapping")

# Communication between MPI processes (bidirectional)
dot.edge("MPI1", "MPI2", label="", style="dashed", color="#87CEEB", dir="both", constraint="false")
dot.edge("MPI2", "MPI3", label="", style="dashed", color="#87CEEB", dir="both", constraint="false")

# 6. Fabric Init and Execution (each CP to its Device)
dot.edge("CP1", "Device1", label="6. Fabric Init\nand Execution", color="#8B008B")
dot.edge("CP2", "Device2", label="Fabric Init\nand Execution", color="#8B008B")
dot.edge("CP3", "Device3", label="Fabric Init\nand Execution", color="#8B008B")

# Communication between Control Planes (bidirectional)
dot.edge("CP1", "CP2", label="", style="dashed", color="#DDA0DD", dir="both", constraint="false")
dot.edge("CP2", "CP3", label="", style="dashed", color="#DDA0DD", dir="both", constraint="false")

# Communication between TT Devices (bidirectional)
dot.edge("Device1", "Device2", label="", style="dashed", color="#F0E68C", dir="both", constraint="false")
dot.edge("Device2", "Device3", label="", style="dashed", color="#F0E68C", dir="both", constraint="false")

# Render
dot.render("diagram1_current_flow", cleanup=True)
print("Generated diagram1_current_flow.png")
