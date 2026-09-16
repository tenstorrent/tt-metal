import { fetchDecoded, loadDecodedFile } from "./load.js";
import { FabricMap } from "./map.js";
import {
  buildModel,
  endpointKey,
  endpointLabel,
  filterRouters,
  provenanceHosts,
} from "./model.js";
import { renderDossier } from "./dossier.js";

const elements = {
  fileInput: document.querySelector("#file-input"),
  fixtureSelect: document.querySelector("#fixture-select"),
  fixtureLabel: document.querySelector(".fixture-select"),
  dropZone: document.querySelector("#drop-zone"),
  emptyState: document.querySelector("#empty-state"),
  session: document.querySelector("#session"),
  sourceName: document.querySelector("#source-name"),
  runTitle: document.querySelector("#run-title"),
  runMeta: document.querySelector("#run-meta"),
  hostMeta: document.querySelector("#host-meta"),
  coverageStrip: document.querySelector("#coverage-strip"),
  modelSummary: document.querySelector("#model-summary"),
  mapRoot: document.querySelector("#map-root"),
  mapRecenter: document.querySelector("#map-recenter"),
  viewMesh: document.querySelector("#view-mesh"),
  viewChip: document.querySelector("#view-chip"),
  mapCrumb: document.querySelector("#map-crumb"),
  dossier: document.querySelector("#dossier"),
  mapPane: document.querySelector(".map-pane"),
  routerBrowser: document.querySelector(".router-browser"),
  routerFilter: document.querySelector("#router-filter"),
  filterCount: document.querySelector("#filter-count"),
  routerList: document.querySelector("#router-list"),
  expertRaw: document.querySelector("#expert-raw"),
  errorBanner: document.querySelector("#error-banner"),
  errorMessage: document.querySelector("#error-message"),
  dismissError: document.querySelector("#dismiss-error"),
};

const state = {
  model: null,
  selectedKey: null,
  chipKey: null,
};

const wideLayout = window.matchMedia("(min-width: 781px)");

function syncRailHeight() {
  if (!wideLayout.matches) {
    elements.routerBrowser.style.height = "";
    return;
  }
  elements.routerBrowser.style.height = `${elements.mapPane.offsetHeight}px`;
}

const map = new FabricMap(elements.mapRoot, {
  onSelect(key) {
    selectRouter(key);
  },
  onDrill(chip) {
    state.chipKey = chip ? chip.key : null;
    elements.viewMesh.classList.toggle("active", !chip);
    elements.viewChip.classList.toggle("active", Boolean(chip));
    elements.viewChip.disabled = !chip;
    elements.viewChip.textContent = chip ? `Chip ${chip.meshId}:${chip.chipId}` : "Chip";
    elements.mapCrumb.textContent = chip ? `Chip ${chip.meshId}:${chip.chipId}` : "Mesh";
    syncRailHeight();
  },
});

if (typeof ResizeObserver !== "undefined") {
  new ResizeObserver(() => syncRailHeight()).observe(elements.mapPane);
}
wideLayout.addEventListener("change", () => syncRailHeight());

function coverageState(name, value) {
  if (name === "ok") {
    return "ok";
  }
  if (name === "routers_total" || name === "captured") {
    return "";
  }
  return Number(value) > 0 ? (name === "reset" || name === "unreadable" ? "danger" : "warning") : "";
}

function renderCoverage(coverage) {
  const metrics = [
    ["routers_total", "Routers"],
    ["captured", "Captured"],
    ["ok", "OK"],
    ["unreadable", "Unreadable"],
    ["reset", "Reset"],
    ["torn", "Torn"],
    ["unknown", "Unknown"],
    ["unsupported", "Unsupported"],
    ["not_captured", "Missing"],
    ["identity_mismatch", "Identity mismatch"],
    ["manifest_sha_unverified", "Unverified manifest"],
  ];
  elements.coverageStrip.replaceChildren(
    ...metrics.map(([name, label]) => {
      const card = document.createElement("div");
      card.className = "metric";
      card.dataset.state = coverageState(name, coverage[name]);

      const title = document.createElement("span");
      title.className = "metric-label";
      title.textContent = label;

      const value = document.createElement("span");
      value.className = "metric-value";
      value.textContent = coverage[name] ?? "—";
      card.append(title, value);
      return card;
    }),
  );
}

function selectedRouter() {
  return state.model?.byEndpoint.get(state.selectedKey) || null;
}

function renderSelection() {
  renderDossier(elements.dossier, state.model, selectedRouter(), { expert: elements.expertRaw.checked });
}

function selectRouter(key) {
  state.selectedKey = key;
  const router = selectedRouter();
  if (router) {
    state.chipKey = `${router.id.mesh_id}:${router.id.chip_id}`;
  }
  map.setSelection(key);
  renderRouterList();
  renderSelection();
}

function showMesh() {
  map.popDrill();
}

function showChip() {
  if (!state.chipKey) {
    return;
  }
  map.drillToChip(state.chipKey);
}

function matchingKeys() {
  return new Set(filterRouters(state.model, elements.routerFilter.value).map((router) => endpointKey(router.id)));
}

function routerRow(router) {
  const key = endpointKey(router.id);
  const row = document.createElement("button");
  row.type = "button";
  row.className = `router-row${key === state.selectedKey ? " selected" : ""}`;

  const endpoint = document.createElement("span");
  endpoint.className = "endpoint";
  endpoint.textContent = `${endpointLabel(router.id)} · ${router.direction || "—"}`;

  const status = document.createElement("span");
  status.className = `status dot ${router.capture?.status || "unknown"}`;
  status.title = router.capture?.status || "unknown";

  row.append(endpoint, status);
  row.addEventListener("click", () => {
    selectRouter(key);
  });
  return row;
}

function renderRouterList() {
  const routers = filterRouters(state.model, elements.routerFilter.value);
  elements.filterCount.textContent = `${routers.length}/${state.model.routers.length}`;
  map.setFilterKeys(elements.routerFilter.value.trim() ? matchingKeys() : null);
  if (routers.length === 0) {
    const empty = document.createElement("p");
    empty.className = "empty-list";
    empty.textContent = "No routers match this filter.";
    elements.routerList.replaceChildren(empty);
    return;
  }
  elements.routerList.replaceChildren(...routers.map(routerRow));
}

function renderSession(decoded, sourceName) {
  const model = buildModel(decoded);
  state.model = model;
  state.selectedKey = null;

  elements.sourceName.textContent = sourceName;
  elements.runTitle.textContent = `${decoded.run.arch} · ${decoded.run.fabric_config}`;
  elements.runMeta.textContent = [
    decoded.fabric_context.topology,
    decoded.fabric_context.packet_header_type,
    `decoded ${decoded.generated_at}`,
  ]
    .filter(Boolean)
    .join(" · ");
  const hosts = provenanceHosts(decoded);
  elements.hostMeta.textContent = hosts.length
    ? `Capture provenance: ${hosts.join(", ")}`
    : "Capture host provenance unavailable";
  elements.modelSummary.textContent =
    `${decoded.topology.meshes.length} mesh${decoded.topology.meshes.length === 1 ? "" : "es"}, ` +
    `${model.chips.size} chips, ${model.routers.length} routers, ${model.links.length} directed links`;

  renderCoverage(decoded.coverage);
  elements.routerFilter.value = "";
  state.chipKey = null;
  elements.viewMesh.classList.add("active");
  elements.viewChip.classList.remove("active");
  elements.viewChip.disabled = true;
  elements.viewChip.textContent = "Chip";
  elements.mapCrumb.textContent = "Mesh";
  map.setModel(model);
  renderRouterList();
  renderSelection();
  elements.emptyState.hidden = true;
  elements.session.hidden = false;
  syncRailHeight();
}

function showError(error) {
  elements.errorMessage.textContent = error instanceof Error ? error.message : String(error);
  elements.errorBanner.hidden = false;
}

async function openDecoded(promise, fallbackName) {
  elements.errorBanner.hidden = true;
  try {
    const loaded = await promise;
    renderSession(loaded.decoded, loaded.sourceName || fallbackName);
  } catch (error) {
    showError(error);
  }
}

async function openFile(file) {
  try {
    await openDecoded(loadDecodedFile(file));
  } finally {
    elements.fileInput.value = "";
  }
}

async function loadFixtureIndex() {
  if (!elements.fixtureSelect || location.protocol === "file:") {
    return;
  }
  try {
    const response = await fetch("fixtures/index.json", { cache: "no-store" });
    if (!response.ok) {
      return;
    }
    const index = await response.json();
    if (!Array.isArray(index) || index.length === 0) {
      return;
    }
    for (const entry of index) {
      const option = document.createElement("option");
      option.value = `fixtures/${entry.file}`;
      option.textContent = entry.title || entry.file;
      elements.fixtureSelect.append(option);
    }
    elements.fixtureLabel.hidden = false;
  } catch {
    /* file:// and missing indexes stay on the picker */
  }
}

elements.fixtureSelect?.addEventListener("change", () => {
  const url = elements.fixtureSelect.value;
  if (url) {
    openDecoded(fetchDecoded(url));
  }
});

elements.fileInput.addEventListener("change", () => {
  const [file] = elements.fileInput.files;
  if (file) {
    openFile(file);
  }
});

for (const eventName of ["dragenter", "dragover"]) {
  elements.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropZone.classList.add("dragging");
  });
}

for (const eventName of ["dragleave", "drop"]) {
  elements.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropZone.classList.remove("dragging");
  });
}

elements.dropZone.addEventListener("drop", (event) => {
  const [file] = event.dataTransfer.files;
  if (file) {
    openFile(file);
  } else {
    showError(new Error("Drop one decoded JSON file"));
  }
});

elements.expertRaw.addEventListener("change", () => {
  if (state.model) {
    renderSelection();
  }
});
elements.routerFilter.addEventListener("input", () => {
  if (state.model) {
    renderRouterList();
  }
});
elements.viewMesh.addEventListener("click", () => {
  showMesh();
});
elements.viewChip.addEventListener("click", () => {
  showChip();
});
elements.mapRecenter.addEventListener("click", () => {
  map.fitView();
});
elements.dismissError.addEventListener("click", () => {
  elements.errorBanner.hidden = true;
});

loadFixtureIndex();

window.addEventListener("keydown", (event) => {
  if (!state.model) {
    return;
  }
  if (event.key === "Escape") {
    map.popDrill();
  } else if (event.key === "0") {
    map.fitView();
  } else if (event.key === "+" || event.key === "=") {
    map.zoom(0.9);
  } else if (event.key === "-" || event.key === "_") {
    map.zoom(1.1);
  }
});
