// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Wires the page together: lists captures from the server or a picked folder, loads the chosen one, and switches
// between the load page and the capture's panels.

import { buildCaptureModel } from "./capture_model.js";
import { createChipView } from "./chip_view.js";
import { collectDroppedFiles, collectPickedFiles, createFolderSource, createServedSource } from "./data_sources.js";
import { CaptureLoadError, loadCapture } from "./load_capture.js";
import { clearLoadStatus, renderCaptureList, renderLoadError, renderLoadProgress } from "./load_view.js";
import { createHardwareView } from "./hardware_view.js";
import { createRouterView } from "./router_view.js";
import { renderSummary } from "./summary_view.js";

const PAGE_TITLE = "Fabric Debug Visualizer";

const page = {
    loadView: document.getElementById("load-view"),
    captureView: document.getElementById("capture-view"),
    openOtherButton: document.getElementById("open-other"),
    captureList: document.getElementById("capture-list"),
    loadStatus: document.getElementById("load-status"),
    dropZone: document.getElementById("drop-zone"),
    folderInput: document.getElementById("folder-input"),
    summary: document.getElementById("summary"),
    hardwarePanel: document.getElementById("hardware-panel"),
    chipPanel: document.getElementById("chip-panel"),
    routerPanel: document.getElementById("router-panel"),
};

const state = {
    loading: false,
    // The capture on show, its views, and what's selected in them.
    model: null,
    hardwareView: null,
    chipView: null,
    routerView: null,
    selectedChipPath: null,
    selectedRouterPath: null,
};

function scrollToPanel(panel) {
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    panel.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
}

// Selects a chip in the hardware and chip views, and closes the router view. A click in the hardware view scrolls down
// to the chip view, which opens below it.
function selectChip(chipPath, { scrollToChipView = false } = {}) {
    state.selectedChipPath = chipPath;
    state.selectedRouterPath = null;
    state.hardwareView.showSelectedChip(chipPath);
    state.chipView.showChip(chipPath);
    state.routerView.showRouter(null);
    if (chipPath && scrollToChipView) scrollToPanel(page.chipPanel);
}

// Selects a router, and its chip when that's another chip, in every view, and scrolls to the router view below the
// chip view. Null closes the router view and keeps the chip.
function selectRouter(routerPath) {
    const router = routerPath ? state.model.routersByPath.get(routerPath) : null;
    if (router && router.chip.path !== state.selectedChipPath) selectChip(router.chip.path);
    state.selectedRouterPath = router ? routerPath : null;
    state.chipView.showSelectedRouter(state.selectedRouterPath);
    state.routerView.showRouter(state.selectedRouterPath);
    if (router) scrollToPanel(page.routerPanel);
}

// Draws a slot's packet path on the hardware view and scrolls up to it; the view's banner scrolls back down.
function showPacketPath(shown) {
    state.hardwareView.showPath({ ...shown, onBack: () => scrollToPanel(page.routerPanel) });
    scrollToPanel(page.hardwarePanel);
}

function showLoadView() {
    page.loadView.hidden = false;
    page.captureView.hidden = true;
    page.openOtherButton.hidden = true;
    document.title = PAGE_TITLE;
}

function showCaptureView(model) {
    state.model = model;
    state.selectedChipPath = null;
    state.selectedRouterPath = null;
    renderSummary(page.summary, model);
    state.hardwareView?.detach();
    state.hardwareView = createHardwareView(page.hardwarePanel, model, {
        onSelectChip: (chipPath) => selectChip(chipPath, { scrollToChipView: true }),
    });
    state.chipView = createChipView(page.chipPanel, model, {
        onSelectChip: (chipPath) => selectChip(chipPath),
        onSelectRouter: (routerPath) => selectRouter(routerPath),
    });
    state.routerView = createRouterView(page.routerPanel, model, {
        onSelectRouter: (routerPath) => selectRouter(routerPath),
        onShowPath: (shown) => showPacketPath(shown),
    });
    page.loadView.hidden = true;
    page.captureView.hidden = false;
    page.openOtherButton.hidden = false;
    document.title = `${model.capture.capture_id} · ${PAGE_TITLE}`;
    window.scrollTo(0, 0);
}

async function openRun(source, capture, run) {
    if (state.loading) return;
    state.loading = true;
    page.loadView.classList.add("busy");
    try {
        const loadedCapture = await loadCapture(source, capture, run, (progress) => renderLoadProgress(page.loadStatus, progress));
        // Drop the previous capture before building the next model, so both aren't held at once.
        state.model = null;
        state.hardwareView?.detach();
        state.hardwareView = null;
        state.chipView = null;
        state.routerView = null;
        showCaptureView(buildCaptureModel(loadedCapture));
        clearLoadStatus(page.loadStatus);
    } catch (error) {
        renderLoadError(page.loadStatus, error instanceof CaptureLoadError ? error.message : `Couldn't load the capture: ${error}`);
        if (!(error instanceof CaptureLoadError)) console.error(error);
    } finally {
        state.loading = false;
        page.loadView.classList.remove("busy");
    }
}

// Lists the source's captures and opens the only one straight away when there's exactly one run to open.
async function showSource(source, heading) {
    clearLoadStatus(page.loadStatus);
    let listing;
    try {
        listing = await source.listCaptures();
    } catch (error) {
        page.captureList.replaceChildren();
        renderLoadError(page.loadStatus, `Couldn't list the captures: ${error.message}`);
        return;
    }
    renderCaptureList(page.captureList, listing, heading, (capture, run) => openRun(source, capture, run));
    const runs = listing.captures.flatMap((capture) => capture.runs.map((run) => ({ capture, run })));
    if (runs.length === 1) await openRun(source, runs[0].capture, runs[0].run);
}

function openPickedFolder(folderName, pickedFiles) {
    if (state.loading) return;
    showLoadView();
    showSource(createFolderSource(folderName, pickedFiles), "Captures in the folder you opened");
}

page.openOtherButton.addEventListener("click", showLoadView);

page.folderInput.addEventListener("change", () => {
    const pickedFiles = collectPickedFiles(page.folderInput.files);
    const firstPath = page.folderInput.files[0]?.webkitRelativePath ?? "";
    page.folderInput.value = "";
    openPickedFolder(firstPath.split("/")[0] || "the folder", pickedFiles);
});

page.dropZone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        page.folderInput.click();
    }
});
for (const eventName of ["dragenter", "dragover"]) {
    page.dropZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        page.dropZone.classList.add("dragging");
    });
}
// A folder dropped beside the drop zone would otherwise make the browser navigate to it.
for (const eventName of ["dragover", "drop"]) window.addEventListener(eventName, (event) => event.preventDefault());
page.dropZone.addEventListener("dragleave", () => page.dropZone.classList.remove("dragging"));
page.dropZone.addEventListener("drop", async (event) => {
    event.preventDefault();
    page.dropZone.classList.remove("dragging");
    const { folderName, pickedFiles } = await collectDroppedFiles(event.dataTransfer);
    openPickedFolder(folderName, pickedFiles);
});

showSource(createServedSource(), "Captures the server found");
