import { captureAppearance, linkAppearance, stallStroke } from "./color.js";
import { renderCardinal } from "./chip.js";
import { layoutModel, PORT } from "./layout.js";

const SVG_NS = "http://www.w3.org/2000/svg";

function svg(name, attributes = {}) {
  const node = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attributes)) {
    if (value !== null && value !== undefined) {
      node.setAttribute(key, String(value));
    }
  }
  return node;
}

function viewBoxString(box) {
  return `${box.x} ${box.y} ${box.w} ${box.h}`;
}

function fitBounds(bounds) {
  return {
    x: bounds.minX,
    y: bounds.minY,
    w: Math.max(1, bounds.maxX - bounds.minX),
    h: Math.max(1, bounds.maxY - bounds.minY),
  };
}

export class FabricMap {
  constructor(root, { onSelect, onDrill } = {}) {
    this.root = root;
    this.onSelect = onSelect;
    this.onDrill = onDrill;
    this.model = null;
    this.layout = null;
    this.matching = null;
    this.selectedKey = null;
    this.drillChipKey = null;
    this.viewBox = { x: 0, y: 0, w: 400, h: 300 };
    this.drag = null;
    this.svg = null;
    this.root.replaceChildren();
  }

  setModel(model) {
    this.model = model;
    this.layout = layoutModel(model);
    this.drillChipKey = null;
    this.selectedKey = null;
    this.matching = null;
    this.viewBox = fitBounds(this.layout.bounds);
    this.render();
  }

  setFilterKeys(keys) {
    this.matching = keys;
    this.render();
  }

  setSelection(key) {
    this.selectedKey = key;
    this.render();
  }

  popDrill() {
    if (this.drillChipKey) {
      this.fitView();
      return true;
    }
    return false;
  }

  fitView() {
    if (!this.layout) {
      return false;
    }
    this.drillChipKey = null;
    this.viewBox = fitBounds(this.layout.bounds);
    this.render();
    this.onDrill?.(null);
    return true;
  }

  zoom(factor, origin) {
    if (!this.layout) {
      return;
    }
    const box = this.viewBox;
    const nextW = Math.max(40, box.w * factor);
    const nextH = Math.max(40, box.h * factor);
    const cx = origin ? origin.x : box.x + box.w / 2;
    const cy = origin ? origin.y : box.y + box.h / 2;
    this.viewBox = {
      x: cx - ((cx - box.x) / box.w) * nextW,
      y: cy - ((cy - box.y) / box.h) * nextH,
      w: nextW,
      h: nextH,
    };
    this.svg?.setAttribute("viewBox", viewBoxString(this.viewBox));
  }

  drillToChip(chipKey) {
    const chip = this.layout.chips.find((entry) => entry.key === chipKey);
    if (!chip) {
      return;
    }
    this.drillChipKey = chipKey;
    const endpoints = chip.chip.routers || [];
    const selectedOnChip = endpoints.some(
      (endpoint) => `${endpoint.mesh_id}:${endpoint.chip_id}:${endpoint.eth_chan}` === this.selectedKey,
    );
    if (!selectedOnChip && endpoints.length) {
      const first = endpoints[0];
      this.selectedKey = `${first.mesh_id}:${first.chip_id}:${first.eth_chan}`;
    }
    this.render();
    this.onDrill?.(chip);
    if (this.selectedKey) {
      this.onSelect?.(this.selectedKey, null);
    }
  }

  pointerToSvg(event) {
    const point = this.svg.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    return point.matrixTransform(this.svg.getScreenCTM().inverse());
  }

  render() {
    if (!this.layout) {
      this.root.replaceChildren();
      return;
    }
    if (this.drillChipKey) {
      this.svg = null;
      renderCardinal(
        this.root,
        this.model,
        this.layout,
        this.drillChipKey,
        this.selectedKey,
        (key) => {
          this.selectedKey = key;
          this.render();
          this.onSelect?.(key, null);
        },
      );
      return;
    }
    const svgNode = svg("svg", {
      class: "fabric-map",
      viewBox: viewBoxString(this.viewBox),
      preserveAspectRatio: "xMidYMid meet",
      role: "img",
      "aria-label": "Fabric topology map",
    });
    const defs = svg("defs");
    const pattern = svg("pattern", {
      id: "torn-hatch",
      width: 6,
      height: 6,
      patternUnits: "userSpaceOnUse",
      patternTransform: "rotate(45)",
    });
    pattern.append(
      svg("rect", { width: 6, height: 6, fill: "#3a3118" }),
      svg("line", { x1: 0, y1: 0, x2: 0, y2: 6, stroke: "#f5c451", "stroke-width": 2 }),
    );
    defs.append(pattern);
    svgNode.append(defs);

    const wraps = svg("g", { class: "layer-wraps" });
    const straights = svg("g", { class: "layer-links" });
    const chipLayer = svg("g", { class: "layer-chips" });
    const portLayer = svg("g", { class: "layer-ports" });

    const matching = this.matching;
    const selected = this.selectedKey;
    const incident = new Set();
    if (selected) {
      for (const drawn of this.layout.links) {
        if (drawn.srcKey === selected || drawn.dstKey === selected) {
          incident.add(drawn.index);
        }
      }
    }

    for (const chip of this.layout.chips) {
      const routers = chip.chip.routers || [];
      const statuses = routers.map((endpoint) => this.model.byEndpoint.get(`${endpoint.mesh_id}:${endpoint.chip_id}:${endpoint.eth_chan}`)?.capture?.status);
      const status = statuses.includes("reset")
        ? "reset"
        : statuses.includes("torn")
          ? "torn"
          : statuses.includes("unreadable")
            ? "unreadable"
            : statuses.every((value) => value === "not_captured" || value === undefined)
              ? "not_captured"
              : statuses.every((value) => value === "ok" || value === undefined)
                ? "ok"
                : "unknown";
      const look = captureAppearance(chip.unplaced ? "unknown" : status);
      const dimmed = matching instanceof Set && !routers.some((endpoint) => matching.has(`${endpoint.mesh_id}:${endpoint.chip_id}:${endpoint.eth_chan}`));
      const node = svg("rect", {
        class: `chip${look.outline ? " reset" : ""}${dimmed ? " dimmed" : ""}`,
        x: chip.x,
        y: chip.y,
        width: chip.w,
        height: chip.h,
        rx: 8,
        fill: look.hatch ? "url(#torn-hatch)" : look.fill,
        stroke: look.stroke,
        "stroke-width": look.outline ? 3 : 1.25,
        "data-chip": chip.key,
      });
      node.addEventListener("click", (event) => {
        event.stopPropagation();
        this.drillToChip(chip.key);
      });
      const label = svg("text", {
        class: "chip-label",
        x: chip.x + chip.w / 2,
        y: chip.y + chip.h / 2 + 4,
        "text-anchor": "middle",
      });
      label.textContent = `${chip.meshId}:${chip.chipId}`;
      chipLayer.append(node, label);
    }

    for (const drawn of this.layout.links) {
      const appearance = linkAppearance(drawn.link);
      const dimmed = matching instanceof Set && !matching.has(drawn.srcKey) && (!drawn.dstKey || !matching.has(drawn.dstKey));
      const highlight = incident.has(drawn.index);
      const attributes = {
        class: `link${appearance.intermesh ? " intermesh" : ""}${appearance.torn ? " torn" : ""}${dimmed ? " dimmed" : ""}${highlight ? " selected" : ""}`,
        fill: "none",
        stroke: appearance.stroke,
        "stroke-width": highlight ? 3 : appearance.wrap ? 2 : 1.6,
        "stroke-dasharray": appearance.intermesh ? "6 4" : null,
        "data-src": drawn.srcKey,
      };
      let node;
      if (drawn.wrap && drawn.control) {
        node = svg("path", {
          ...attributes,
          d: `M ${drawn.x1} ${drawn.y1} Q ${drawn.control.cx} ${drawn.control.cy} ${drawn.x2} ${drawn.y2}`,
        });
        wraps.append(node);
      } else {
        node = svg("line", { ...attributes, x1: drawn.x1, y1: drawn.y1, x2: drawn.x2, y2: drawn.y2 });
        straights.append(node);
      }
      node.addEventListener("click", (event) => {
        event.stopPropagation();
        this.selectedKey = drawn.srcKey;
        this.render();
        this.onSelect?.(drawn.srcKey, drawn.link);
      });
      if (appearance.crossHost) {
        const mark = svg("circle", {
          class: "cross-host",
          cx: (drawn.x1 + drawn.x2) / 2,
          cy: (drawn.y1 + drawn.y2) / 2,
          r: 3.5,
          fill: "#f5c451",
        });
        straights.append(mark);
      }
    }

    for (const port of this.layout.ports) {
      const look = captureAppearance(port.router.capture?.status);
      const dimmed = matching instanceof Set && !matching.has(port.key);
      const selectedPort = port.key === selected;
      const stall = port.router.capture?.status === "ok" ? stallStroke(port.router.stall_score) : look.stroke;
      const node = svg("circle", {
        class: `port${look.outline ? " reset" : ""}${dimmed ? " dimmed" : ""}${selectedPort ? " selected" : ""}`,
        cx: port.x,
        cy: port.y,
        r: selectedPort ? PORT / 2 + 2 : PORT / 2,
        fill: look.hatch ? "url(#torn-hatch)" : look.fill,
        stroke: stall,
        "stroke-width": look.outline || selectedPort ? 2.5 : 1.5,
        "data-router": port.key,
      });
      node.addEventListener("click", (event) => {
        event.stopPropagation();
        this.selectedKey = port.key;
        this.render();
        this.onSelect?.(port.key, null);
      });
      portLayer.append(node);
    }

    svgNode.append(wraps, straights, chipLayer, portLayer);
    svgNode.addEventListener("wheel", (event) => {
      event.preventDefault();
      const point = this.pointerToSvg(event);
      this.zoom(event.deltaY < 0 ? 0.9 : 1.1, point);
    }, { passive: false });
    svgNode.addEventListener("pointerdown", (event) => {
      if (event.target !== svgNode) {
        return;
      }
      this.drag = {
        clientX: event.clientX,
        clientY: event.clientY,
        box: { ...this.viewBox },
        pointer: event.pointerId,
      };
      svgNode.setPointerCapture(event.pointerId);
    });
    svgNode.addEventListener("pointermove", (event) => {
      if (!this.drag || event.pointerId !== this.drag.pointer) {
        return;
      }
      const scaleX = this.drag.box.w / Math.max(1, svgNode.clientWidth);
      const scaleY = this.drag.box.h / Math.max(1, svgNode.clientHeight);
      this.viewBox = {
        ...this.drag.box,
        x: this.drag.box.x - (event.clientX - this.drag.clientX) * scaleX,
        y: this.drag.box.y - (event.clientY - this.drag.clientY) * scaleY,
      };
      svgNode.setAttribute("viewBox", viewBoxString(this.viewBox));
    });
    svgNode.addEventListener("pointerup", () => {
      this.drag = null;
    });
    this.svg = svgNode;
    this.root.replaceChildren(svgNode);
  }
}
