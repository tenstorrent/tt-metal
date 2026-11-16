// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Zoom and pan for an SVG by changing its viewBox: the wheel or a pinch zooms around the pointer, dragging pans. A
// drag doesn't count as a click, so clicking a chip still works after panning.

const MAXIMUM_ZOOM_IN = 12;
const MAXIMUM_ZOOM_OUT = 2;
const DRAG_THRESHOLD_PIXELS = 4;

// `contentBounds` is { x, y, width, height } in SVG units: what fit() shows.
export function attachZoomPan(svg, contentBounds, { margin = 24 } = {}) {
    const fitted = {
        x: contentBounds.x - margin,
        y: contentBounds.y - margin,
        width: contentBounds.width + 2 * margin,
        height: contentBounds.height + 2 * margin,
    };
    let view = { ...fitted };
    const listeners = new AbortController();
    const signal = listeners.signal;
    const applyView = () => svg.setAttribute("viewBox", `${view.x} ${view.y} ${view.width} ${view.height}`);

    function toSvgPoint(clientX, clientY) {
        return new DOMPoint(clientX, clientY).matrixTransform(svg.getScreenCTM().inverse());
    }

    // factor < 1 zooms in, keeping `point` where it is on screen.
    function zoomAround(point, factor) {
        const width = Math.min(Math.max(view.width * factor, fitted.width / MAXIMUM_ZOOM_IN), fitted.width * MAXIMUM_ZOOM_OUT);
        const scale = width / view.width;
        view = {
            x: point.x - (point.x - view.x) * scale,
            y: point.y - (point.y - view.y) * scale,
            width: view.width * scale,
            height: view.height * scale,
        };
        applyView();
    }

    svg.addEventListener("wheel", (event) => {
        event.preventDefault();
        const pixels = event.deltaY * (event.deltaMode === 1 ? 16 : 1);
        // A trackpad pinch arrives as a wheel event with ctrlKey and small deltas.
        const sensitivity = event.ctrlKey ? 0.01 : 0.0015;
        zoomAround(toSvgPoint(event.clientX, event.clientY), Math.exp(pixels * sensitivity));
    }, { passive: false, signal });

    // Pointers down on the SVG, for dragging with one and pinching with two. Moves are followed on the window, so a
    // drag that leaves the SVG keeps going, and the pointer isn't captured, so a click still reaches the chip under it.
    const pointers = new Map();
    let draggedPixels = 0;

    svg.addEventListener("pointerdown", (event) => {
        if (event.button !== 0) return;
        pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
        if (pointers.size === 1) draggedPixels = 0;
    }, { signal });

    window.addEventListener("pointermove", (event) => {
        const previous = pointers.get(event.pointerId);
        if (!previous) return;
        const current = { x: event.clientX, y: event.clientY };
        if (pointers.size === 1) {
            const pixelsPerUnit = svg.getScreenCTM().a;
            view.x -= (current.x - previous.x) / pixelsPerUnit;
            view.y -= (current.y - previous.y) / pixelsPerUnit;
            applyView();
        } else if (pointers.size === 2) {
            const [other] = [...pointers].filter(([id]) => id !== event.pointerId).map(([, position]) => position);
            const before = Math.hypot(previous.x - other.x, previous.y - other.y);
            const after = Math.hypot(current.x - other.x, current.y - other.y);
            if (before > 0 && after > 0) {
                zoomAround(toSvgPoint((current.x + other.x) / 2, (current.y + other.y) / 2), before / after);
            }
        }
        draggedPixels += Math.hypot(current.x - previous.x, current.y - previous.y);
        pointers.set(event.pointerId, current);
    }, { signal });

    const releasePointer = (event) => pointers.delete(event.pointerId);
    window.addEventListener("pointerup", releasePointer, { signal });
    window.addEventListener("pointercancel", releasePointer, { signal });

    // Swallows the click that ends a drag, before it reaches a chip.
    svg.addEventListener("click", (event) => {
        if (draggedPixels > DRAG_THRESHOLD_PIXELS) {
            event.stopPropagation();
            event.preventDefault();
        }
        draggedPixels = 0;
    }, { capture: true, signal });

    applyView();
    return {
        fit() {
            view = { ...fitted };
            applyView();
        },
        zoomBy(factor) {
            zoomAround({ x: view.x + view.width / 2, y: view.y + view.height / 2 }, factor);
        },
        // Removes every listener, including those on the window.
        detach() {
            listeners.abort();
        },
    };
}
