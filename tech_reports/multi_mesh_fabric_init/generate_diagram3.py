#!/usr/bin/env python3
"""Generate diagram 3: Phase 2 Flow (Fully Automated) - Two Column Layout"""

import graphviz

# Create a new directed graph
dot = graphviz.Digraph(comment="Phase 2 Flow Diagram", format="png")
dot.attr(rankdir="TB", nodesep="0.4", ranksep="0.7", compound="true")
dot.attr("node", fontsize="11", fontname="Arial")
dot.attr("edge", fontsize="10", fontname="Arial")

# --- Nodes ---

dot.node("User", "User", shape="ellipse", style="filled", fillcolor="#FFE5B4", width="1.5")
dot.node("Groupings", "physical_groupings.yaml", shape="note", style="filled", fillcolor="#FFB6C1", fontsize="10")
dot.node(
    "MGD_Input", "mesh_graph_descriptor.textproto", shape="note", style="filled", fillcolor="#FFB6C1", fontsize="10"
)
dot.node("SLURM", "SLURM\nScheduler", shape="box", style="rounded,filled", fillcolor="#90EE90", width="1.5")

with dot.subgraph(name="cluster_FM_System") as fm_sys:
    fm_sys.attr(label="Fabric Manager System", style="filled", color="#E0F0FF", fillcolor="#E0F0FF")
    dot.node("FM", "Fabric Manager\n(Service Daemon)", shape="box", style="filled", fillcolor="#87CEFA", width="1.8")
    dot.node("Topo", "Topology\nMapper", shape="box", style="filled", fillcolor="#B0E0E6", width="1.2")

    # Internal edge
    dot.edge("FM", "Topo", label="Solve", color="#0066CC")
    dot.edge("Topo", "FM", label="Alloc", color="#0066CC")

# Multiple Fabric Agents - grouped
with dot.subgraph(name="cluster_Agents") as agents:
    agents.attr(label="Fabric Agents", style="filled", color="#F0E0FF", fillcolor="#F0E0FF")
    dot.node("Agent1", "Agent 1\n(gRPC)", shape="box", style="rounded,filled", fillcolor="#DDA0DD", width="1.3")
    dot.node("Agent2", "Agent 2\n(gRPC)", shape="box", style="rounded,filled", fillcolor="#DDA0DD", width="1.3")
    dot.node("Agent3", "Agent 3\n(gRPC)", shape="box", style="rounded,filled", fillcolor="#DDA0DD", width="1.3")

# Right column: tt-run and MPI
dot.node("TTRun", "tt-run", shape="box", style="rounded,filled", fillcolor="#87CEEB", width="1.2")

# Multiple MPI Processes - grouped
with dot.subgraph(name="cluster_MPI") as mpi:
    mpi.attr(label="MPI Processes", style="filled", color="#E0F0FF", fillcolor="#E0F0FF")
    dot.node("MPI1", "MPI Process 1", shape="box", style="rounded,filled", fillcolor="#87CEEB", width="1.4")
    dot.node("MPI2", "MPI Process 2", shape="box", style="rounded,filled", fillcolor="#87CEEB", width="1.4")
    dot.node("MPI3", "MPI Process 3", shape="box", style="rounded,filled", fillcolor="#87CEEB", width="1.4")

# Multiple TT Devices - grouped
with dot.subgraph(name="cluster_Devices") as devices:
    devices.attr(label="TT Devices (Unified Kernel)", style="filled", color="#FFF8DC", fillcolor="#FFF8DC")
    dot.node("Device1", "TT Device 1", shape="box", style="filled", fillcolor="#F0E68C", width="1.4")
    dot.node("Device2", "TT Device 2", shape="box", style="filled", fillcolor="#F0E68C", width="1.4")
    dot.node("Device3", "TT Device 3", shape="box", style="filled", fillcolor="#F0E68C", width="1.4")

# --- Strict Vertical Ranks ---

# Rank 1: User
with dot.subgraph() as r1:
    r1.attr(rank="same")
    r1.node("User")

# Rank 2: Groupings and MGD (Groupings on left)
with dot.subgraph() as r2:
    r2.attr(rank="same")
    r2.node("Groupings")
    r2.node("MGD_Input")

# Rank 3: SLURM
with dot.subgraph() as r3:
    r3.attr(rank="same")
    r3.node("SLURM")

# Rank 4: FM
with dot.subgraph() as r4:
    r4.attr(rank="same")
    r4.node("FM")

# Rank 5: Fabric Agents (left only)
with dot.subgraph() as r5:
    r5.attr(rank="same")
    r5.node("Agent1")
    r5.node("Agent2")
    r5.node("Agent3")

# Rank 6: tt-run (right, between agents and devices)
with dot.subgraph() as r6:
    r6.attr(rank="same")
    r6.node("TTRun")

# Rank 7: MPI Processes (right, under tt-run)
with dot.subgraph() as r7:
    r7.attr(rank="same")
    r7.node("MPI1")
    r7.node("MPI2")
    r7.node("MPI3")

# Rank 8: TT Devices (left, aligned with agents, below agents)
with dot.subgraph() as r8:
    r8.attr(rank="same")
    r8.node("Device1")
    r8.node("Device2")
    r8.node("Device3")

# --- Vertical Alignment (Invisible) ---
# Main column: User -> Groupings -> MGD -> SLURM -> FM -> Agents -> Devices
dot.edge("User", "Groupings", style="invis")
dot.edge("Groupings", "MGD_Input", style="invis")
dot.edge("MGD_Input", "SLURM", style="invis")
dot.edge("SLURM", "FM", style="invis")
dot.edge("FM", "Agent2", style="invis")  # Center agent
dot.edge("Agent2", "Device2", style="invis")  # Center device

# Right column: tt-run -> MPI Processes (separate group, keep vertical spacing)
dot.edge("TTRun", "MPI2", style="invis")  # tt-run above center MPI (maintains vertical spacing)

# Horizontal alignment within groups
dot.edge("Groupings", "MGD_Input", style="invis")
dot.edge("Agent1", "Agent2", style="invis")
dot.edge("Agent2", "Agent3", style="invis")
dot.edge("MPI1", "MPI2", style="invis")
dot.edge("MPI2", "MPI3", style="invis")
dot.edge("Device1", "Device2", style="invis")
dot.edge("Device2", "Device3", style="invis")

# Separate left and right columns (push right column away)
dot.edge("Agent3", "TTRun", style="invis")  # Separate left and right columns

# --- Edges ---

# 1. Setup Flow (Top to Bottom) - Clean vertical flow
dot.edge("User", "MGD_Input", label="1. Create", style="dashed")
dot.edge("User", "SLURM", label="2. srun w/ MGD")
dot.edge("MGD_Input", "SLURM", label="Receive", style="dashed", constraint="false")
dot.edge("Groupings", "SLURM", label="Pass", style="dashed", constraint="false")
dot.edge("SLURM", "FM", label="3. Query")
dot.edge("MGD_Input", "FM", label="Parse for mapping", constraint="false")
dot.edge("Groupings", "FM", label="Grouping info", style="dashed", constraint="false")

# FM bidirectional gRPC communication with all Fabric Agents
dot.edge("FM", "Agent1", label="4. gRPC", dir="both", color="#006400", penwidth="2.0")
dot.edge("FM", "Agent2", label="gRPC", dir="both", color="#006400", penwidth="2.0")
dot.edge("FM", "Agent3", label="gRPC", dir="both", color="#006400", penwidth="2.0")

# Each Agent sets up its Device (straight down, perfectly aligned)
dot.edge("Agent1", "Device1", label="5. Setup Device", color="#006400", penwidth="2.0")
dot.edge("Agent2", "Device2", label="Setup Device", color="#006400", penwidth="2.0")
dot.edge("Agent3", "Device3", label="Setup Device", color="#006400", penwidth="2.0")

# 2. Execution Flow - Right column
dot.edge("User", "TTRun", label="6. Exec", constraint="false", style="bold")
dot.edge(
    "TTRun", "FM", label="7. Handshake", dir="both", color="#8B008B", penwidth="2.0", constraint="false", style="bold"
)

# tt-run launches multiple MPI processes (straight down, aligned)
dot.edge("TTRun", "MPI1", label="8. Launch", color="#87CEEB", penwidth="2.0")
dot.edge("TTRun", "MPI2", label="Launch", color="#87CEEB", penwidth="2.0")
dot.edge("TTRun", "MPI3", label="Launch", color="#87CEEB", penwidth="2.0")

# Each MPI process runs directly on its Device (cross from right to left)
dot.edge("MPI1", "Device1", label="9. Run Directly", color="#FF4500", penwidth="2.0", constraint="false")
dot.edge("MPI2", "Device2", label="Run Directly", color="#FF4500", penwidth="2.0", constraint="false")
dot.edge("MPI3", "Device3", label="Run Directly", color="#FF4500", penwidth="2.0", constraint="false")

# Communication between TT Devices (bidirectional, horizontal, clean)
dot.edge("Device1", "Device2", label="", style="dashed", color="#DAA520", dir="both", penwidth="1.5")
dot.edge("Device2", "Device3", label="", style="dashed", color="#DAA520", dir="both", penwidth="1.5")

# Return status (clean, minimal crossing)
dot.edge("FM", "SLURM", label="Ready", style="dashed", color="#666666", constraint="false")
dot.edge("SLURM", "User", label="Start", style="dashed", color="#666666", constraint="false")

# Render
dot.render("diagram3_phase2_flow", cleanup=True)
print("Generated diagram3_phase2_flow.png")
