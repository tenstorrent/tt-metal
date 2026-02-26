import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def draw_model_graph(model, max_depth=4, output_file="model_graph.png"):
    """Create a clean tree visualization showing module nesting."""

    # Build tree structure
    class Node:
        def __init__(self, name, cls_name, depth):
            self.name = name
            self.cls_name = cls_name
            self.depth = depth
            self.children = []
            self.y_pos = 0

    root_nodes = []
    seen_modules = set()  # Track modules to avoid duplicates

    def build_tree(module, parent=None, depth=0, parent_path=""):
        if depth >= max_depth:
            return None

        # Use id() to track actual module instances, not paths
        module_id = id(module)

        children_list = list(module.named_children())

        # Group consecutive children with same class type
        i = 0
        while i < len(children_list):
            name, child = children_list[i]
            child_id = id(child)

            # Skip if already processed
            if child_id in seen_modules:
                i += 1
                continue

            child_cls = child.__class__.__name__

            # Look ahead for consecutive children of the same type
            consecutive_count = 1
            consecutive_names = [name]
            j = i + 1

            while j < len(children_list):
                next_name, next_child = children_list[j]
                if next_child.__class__.__name__ == child_cls and id(next_child) not in seen_modules:
                    consecutive_count += 1
                    consecutive_names.append(next_name)
                    seen_modules.add(id(next_child))
                    j += 1
                else:
                    break

            seen_modules.add(child_id)

            # If we found repeated modules, collapse them
            if consecutive_count > 2:
                # Create a collapsed node
                display_name = f"{consecutive_names[0]}-{consecutive_names[-1]} (x{consecutive_count})"
                node = Node(display_name, child_cls, depth)

                if parent:
                    parent.children.append(node)
                else:
                    root_nodes.append(node)

                # Only recurse into the first one to show the structure
                build_tree(
                    child,
                    node,
                    depth + 1,
                    f"{parent_path}.{consecutive_names[0]}" if parent_path else consecutive_names[0],
                )
            else:
                # Regular single node
                full_path = f"{parent_path}.{name}" if parent_path else name
                node = Node(name, child_cls, depth)

                if parent:
                    parent.children.append(node)
                else:
                    root_nodes.append(node)

                build_tree(child, node, depth + 1, full_path)

            i = j if consecutive_count > 2 else i + 1

    build_tree(model)

    print(f"Found {len(root_nodes)} root modules, {len(seen_modules)} unique modules")

    if not root_nodes:
        print("No modules to visualize - model may have no children or max_depth too low")
        return

    # Calculate positions using proper tree layout
    def assign_positions(nodes, y_start=0):
        y = y_start
        for node in nodes:
            if node.children:
                # Position children first
                child_start = y
                y = assign_positions(node.children, y)
                # Position parent at midpoint of children
                node.y_pos = (child_start + y - 1) / 2
            else:
                node.y_pos = y
                y += 1
        return y

    total_height = assign_positions(root_nodes)

    print(f"Tree height: {total_height}")

    if total_height == 0:
        print("Warning: total_height is 0, setting to 1")
        total_height = 1

    # Collect all nodes in a flat list for drawing
    all_nodes = []

    def collect_all(nodes):
        for node in nodes:
            all_nodes.append(node)
            collect_all(node.children)

    collect_all(root_nodes)

    print(f"Drawing {len(all_nodes)} nodes...")

    if not all_nodes:
        print("No nodes to draw!")
        return

    # Create figure with dynamic sizing
    fig_height = max(8, total_height * 0.6)
    fig_width = max(12, max_depth * 4 + 4)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Colors for depth
    colors = [
        "#e8f4f8",
        "#b3e5fc",
        "#81d4fa",
        "#4fc3f7",
        "#29b6f6",
        "#03a9f4",
        "#039be5",
        "#0288d1",
        "#0277bd",
        "#01579b",
        "#014f86",
    ]

    # Draw each node
    for node in all_nodes:
        x = node.depth * 4 + 1
        y = node.y_pos

        # Box dimensions
        box_width = 3.2
        box_height = 0.5

        # Color based on depth
        color = colors[min(node.depth, len(colors) - 1)]

        # Draw box
        box = Rectangle(
            (x, y - box_height / 2),
            box_width,
            box_height,
            facecolor=color,
            edgecolor="#01579b",
            linewidth=1.5,
            alpha=0.9,
        )
        ax.add_patch(box)

        # Add text - module name
        ax.text(
            x + box_width / 2,
            y + 0.08,
            node.cls_name[:40],  # Truncate long class names
            fontsize=7,
            weight="bold",
            va="center",
            ha="center",
            color="#01579b",
        )

        # Class name below
        ax.text(
            x + box_width / 2,
            y - 0.12,
            node.name[:60],  # Truncate long names
            fontsize=5,
            va="center",
            ha="center",
            style="italic",
            color="#0277bd",
        )

        # Draw lines to children
        if node.children:
            child_x = (node.depth + 1) * 4 + 1
            # Horizontal line from node to children column
            ax.plot([x + box_width, child_x], [y, y], color="#0288d1", linewidth=1, alpha=0.6)
            # Vertical lines to each child
            for child in node.children:
                ax.plot([child_x, child_x], [y, child.y_pos], color="#0288d1", linewidth=1, alpha=0.6)

    # Add level labels
    for d in range(max_depth):
        if any(n.depth == d for n in all_nodes):
            ax.text(
                d * 4 + 2.6,
                total_height + 0.5,
                f"L{d}",
                fontsize=8,
                weight="bold",
                ha="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#e1f5fe", edgecolor="#01579b", linewidth=1),
            )

    ax.set_xlim(-0.5, max_depth * 4 + 6)
    ax.set_ylim(-1, total_height + 1.5)
    ax.axis("off")
    ax.set_title("Model Hierarchy", fontsize=12, weight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"✓ Saved to {output_file} ({len(all_nodes)} modules)")
