#!/usr/bin/env python3
"""Generate diagram 2: Phase 1 Flow (FM-Assisted) - Strictly Vertical Layout"""

import graphviz

# Create a new directed graph
dot = graphviz.Digraph(comment="Phase 1 Flow Diagram", format="png")
dot.attr(rankdir="TB", nodesep="0.3", ranksep="0.6", compound="true")
dot.attr("node", fontsize="11", fontname="Arial")
dot.attr("edge", fontsize="10", fontname="Arial")

# --- Nodes ---

dot.node("User", "User", shape="ellipse", style="filled", fillcolor="#FFE5B4", width="1.5")
dot.node("SLURM", "SLURM\nScheduler", shape="box", style="rounded,filled", fillcolor="#90EE90", width="1.5")

with dot.subgraph(name="cluster_FM_System") as fm_sys:
    fm_sys.attr(label="Fabric Manager System", style="filled", color="#E0F0FF", fillcolor="#E0F0FF")
    dot.node("FM", "Fabric Manager", shape="box", style="filled", fillcolor="#87CEFA", width="1.8")
    dot.node("Topo", "Topology\nMapper", shape="box", style="filled", fillcolor="#B0E0E6", width="1.2")

    # Internal edge
    dot.edge("FM", "Topo", label="Map MGD", style="solid", color="#0066CC")
    dot.edge("Topo", "FM", label="Allocation", style="solid", color="#0066CC")

dot.node(
    "MGD_Input", "mesh_graph_descriptor.textproto", shape="note", style="filled", fillcolor="#FFB6C1", fontsize="10"
)
dot.node(
    "GenRankBindings", "generated_rank_bindings.yaml", shape="note", style="filled", fillcolor="#FFB6C1", fontsize="10"
)
dot.node("Rankfile", "rankfile.txt", shape="note", style="filled", fillcolor="#FFB6C1", fontsize="10")

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

# Rank 2: MGD
with dot.subgraph() as r2:
    r2.attr(rank="same")
    r2.node("MGD_Input")

# Rank 3: SLURM
with dot.subgraph() as r3:
    r3.attr(rank="same")
    r3.node("SLURM")

# Rank 4: Fabric Manager
with dot.subgraph() as r4:
    r4.attr(rank="same")
    r4.node("FM")

# Rank 5: Generated Files
with dot.subgraph() as r5:
    r5.attr(rank="same")
    r5.node("GenRankBindings")
    r5.node("Rankfile")

# Rank 6: tt-run
with dot.subgraph() as r6:
    r6.attr(rank="same")
    r6.node("TTRun")

# Rank 7: MPI Processes (Multiple)
with dot.subgraph() as r7:
    r7.attr(rank="same")
    r7.node("MPI1")
    r7.node("MPI2")
    r7.node("MPI3")

# Rank 8: Control Plane (Multiple)
with dot.subgraph() as r8:
    r8.attr(rank="same")
    r8.node("CP1")
    r8.node("CP2")
    r8.node("CP3")

# Rank 9: Devices (Bottom)
with dot.subgraph() as r9:
    r9.attr(rank="same")
    r9.node("Device1")
    r9.node("Device2")
    r9.node("Device3")

# --- Vertical Alignment (Invisible) ---
dot.edge("User", "MGD_Input", style="invis")
dot.edge("MGD_Input", "SLURM", style="invis")
dot.edge("SLURM", "FM", style="invis")
dot.edge("FM", "GenRankBindings", style="invis")
dot.edge("GenRankBindings", "TTRun", style="invis")
dot.edge("TTRun", "MPI1", style="invis")
dot.edge("MPI1", "CP1", style="invis")
dot.edge("CP1", "Device1", style="invis")

# Keep multiple instances aligned
dot.edge("MPI1", "MPI2", style="invis")
dot.edge("MPI2", "MPI3", style="invis")
dot.edge("CP1", "CP2", style="invis")
dot.edge("CP2", "CP3", style="invis")
dot.edge("Device1", "Device2", style="invis")
dot.edge("Device2", "Device3", style="invis")

# --- Edges ---

# 1. Allocation Flow
dot.edge("User", "MGD_Input", label="1. Create", style="dashed")
dot.edge("User", "SLURM", label="2. salloc w/ MGD")
dot.edge("MGD_Input", "SLURM", label="Receive", style="dashed", constraint="false")
dot.edge("SLURM", "FM", label="3. Query")
dot.edge("MGD_Input", "FM", label="Parse for mapping", constraint="false")
dot.edge("FM", "GenRankBindings", label="4. Generate")
dot.edge("FM", "Rankfile", label="Generate")
dot.edge("FM", "SLURM", label="Return", style="dashed", constraint="false")
dot.edge("SLURM", "User", label="Allocated", style="dashed", constraint="false")

# 2. Execution Flow
dot.edge("User", "TTRun", label="5. Exec", constraint="false")
dot.edge("GenRankBindings", "TTRun", label="--rank-binding", color="#C71585", penwidth="2.0")
dot.edge("Rankfile", "TTRun", label="--rankfile", color="#C71585", penwidth="2.0")
dot.edge("MGD_Input", "TTRun", label="Read MGD", color="#C71585", penwidth="2.0")

# 3. Launch (to all MPI processes)
dot.edge("TTRun", "MPI1", label="6. Launch")
dot.edge("TTRun", "MPI2", label="Launch")
dot.edge("TTRun", "MPI3", label="Launch")

# 4. Discovery and Mapping (each MPI to its CP)
dot.edge("MPI1", "CP1", label="7. Discovery\nand Mapping")
dot.edge("MPI2", "CP2", label="Discovery\nand Mapping")
dot.edge("MPI3", "CP3", label="Discovery\nand Mapping")

# Communication between MPI processes (bidirectional)
dot.edge("MPI1", "MPI2", label="", style="dashed", color="#87CEEB", dir="both", constraint="false")
dot.edge("MPI2", "MPI3", label="", style="dashed", color="#87CEEB", dir="both", constraint="false")

# 5. Fabric Init and Execution (each CP to its Device)
dot.edge("CP1", "Device1", label="8. Fabric Init\nand Execution")
dot.edge("CP2", "Device2", label="Fabric Init\nand Execution")
dot.edge("CP3", "Device3", label="Fabric Init\nand Execution")

# Communication between Control Planes (bidirectional)
dot.edge("CP1", "CP2", label="", style="dashed", color="#DDA0DD", dir="both", constraint="false")
dot.edge("CP2", "CP3", label="", style="dashed", color="#DDA0DD", dir="both", constraint="false")

# Communication between TT Devices (bidirectional)
dot.edge("Device1", "Device2", label="", style="dashed", color="#F0E68C", dir="both", constraint="false")
dot.edge("Device2", "Device3", label="", style="dashed", color="#F0E68C", dir="both", constraint="false")

# Render
dot.render("diagram2_phase1_flow", cleanup=True)
print("Generated diagram2_phase1_flow.png")
